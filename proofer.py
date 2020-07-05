from meshparty import meshwork, trimesh_io, mesh_filters
from annotationframeworkclient.frameworkclient import FrameworkClient
import numpy as np
import pandas as pd
import nglui
import re

EP_TAGS = ['nat_end', 'unc_end', 'unc_cont']
BP_TAGS = ['checked']
EP_COLOR = '#BBCC33'
BP_COLOR = '#3322BB'
STOP_COLOR = "#DD1111"


def _query_statebuilder(img_source=None,
                        seg_source=None,
                        base_state=None,
                        selection_layer='select_to_proof',
                        ignore_layer='ignore_beyond',
                        ctr_pt_layer='center_point'):
    layers = []
    if base_state is None:
        img = nglui.statebuilder.ImageLayerConfig(
            name='img', source=img_source, contrast_controls=True)
        seg = nglui.statebuilder.SegmentationLayerConfig(
            name='select_to_proof', source=seg_source)
        layers.append(img)
        layers.append(seg)
    layers.append(nglui.statebuilder.AnnotationLayerConfig(name=ignore_layer))
    layers.append(nglui.statebuilder.AnnotationLayerConfig(name=ctr_pt_layer))
    sb = nglui.statebuilder.StateBuilder(layers, base_state=base_state)
    return sb


def initial_state_url(datastack, return_as='url', base_state=None):
    """ Make a state from which to do generate proofreading
    """
    client = FrameworkClient(datastack)
    sb = _query_statebuilder(client.info.image_source(),
                             client.info.segmentation_source(),
                             base_state=base_state)
    return sb.render_state(return_as=return_as, url_prefix=client.info.info_cache.get(datastack).get('viewer_site'))


def parse_state(state_url, datastack, client):
    m = re.search(client.state.build_neuroglancer_url(r'(\d*)$', ''), state_url)
    if m is None:
        raise ValueError("URL is not formatted as expected")
    state_id = m.groups()[0]
    state = client.state.get_state_json(int(state_id))
    segs, annos = _parse_state(state)
    return segs, annos


def _parse_state(state, seg_layer='select_to_proof', ignore_layer='ignore_beyond', ctr_pt_layer='center_point'):
    segs = None
    for l in state['layers']:
        if l['name'] == seg_layer:
            segs = [int(x) for x in l['segments']]

    annos = {'center_point': [], 'ignore': []}
    for l in state['layers']:
        if l['name'] == ignore_layer:
            for anno in l['annotations']:
                if anno['type'] == 'point':
                    annos['ignore'].append(anno['point'])

    for l in state['layers']:
        if l['name'] == ctr_pt_layer:
            for anno in l['annotations']:
                if anno['type'] == 'point':
                    annos['center_point'].append(anno['point'])

    return segs, annos


def proofreading_statebuilder(client,
                              fixed_ids,
                              bp_tags=BP_TAGS,
                              bp_color=BP_COLOR,
                              ep_tags=EP_TAGS,
                              ep_color=EP_COLOR,
                              stop_color=STOP_COLOR):
    img = nglui.statebuilder.ImageLayerConfig(
        source=client.info.image_source(), contrast_controls=True)
    seg = nglui.statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(), fixed_ids=fixed_ids)
    bp_points = nglui.statebuilder.PointMapper(set_position=True)
    bp_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='branch_points', color=bp_color, mapping_rules=bp_points, array_data=True, tags=bp_tags, active=True)
    sb_bp = nglui.statebuilder.StateBuilder([img, seg, bp_annos])

    ep_points = nglui.statebuilder.PointMapper()
    ep_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='end_points', color=ep_color, mapping_rules=ep_points, array_data=True, tags=ep_tags, active=False)
    sb_ep = nglui.statebuilder.StateBuilder([ep_annos])

    stop_points = nglui.statebuilder.PointMapper()
    stop_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='axon_point', color=stop_color, mapping_rules=ep_points, array_data=True, tags=ep_tags, active=False)
    sb_stop = nglui.statebuilder.StateBuilder([stop_annos])
    return nglui.statebuilder.ChainedStateBuilder([sb_bp, sb_ep, sb_stop])


def _branch_points_ordered(nrn):
    bps = nrn.branch_points
    bps = bps[bps != nrn.root]
    bp_ord = np.argsort(nrn.distance_to_root(bps))
    return nrn.mesh.vertices[bps[bp_ord]]


def _end_points_ordered(nrn):
    eps = nrn.end_points
    ep_ord = np.argsort(nrn.distance_to_root(eps))
    return nrn.mesh.vertices[eps[ep_ord]]


def get_topo_points(client, segs, annos, min_size=1000, voxel_resolution=np.array([4, 4, 40])):
    mm = trimesh_io.MeshMeta(cv_path=client.info.segmentation_source(),
                             map_gs_to_https=True, disk_cache_path='meshes')

    multi_seg = len(segs) > 1

    all_bps = []
    all_eps = []
    for ii, oid in enumerate(segs):
        mesh = mm.mesh(seg_id=oid)

        lcs = mesh_filters.filter_components_by_size(mesh, min_size=min_size)
        meshf = mesh.apply_mask(lcs)

        nrn = meshwork.Meshwork(meshf)
        if len(annos['center_point']) > 0:
            ctr_pt = np.array(annos['center_point']) * np.array([4, 4, 40])
            nrn.skeletonize_mesh(soma_pt=ctr_pt, soma_thresh_distance=15000,
                                 compute_radius=False, collapse_function='sphere')
        else:
            nrn.skeletonize_mesh(compute_radius=False)

        if len(annos['ignore']) > 0:
            anno_df = pd.DataFrame({'pts': annos['ignore']})
            nrn.add_annotations('ignore_beyond', anno_df, point_column='pts')
            ignore_masks = []
            for ii in nrn.anno['ignore_beyond'].mesh_index:
                ignore_masks.append(np.invert(nrn.downstream_of(ii).to_mesh_mask_base))
            for mask in ignore_masks:
                nrn.apply_mask(mask)

        bps = _branch_points_ordered(nrn) / voxel_resolution
        all_bps.append(bps)

        eps = _end_points_ordered(nrn) / voxel_resolution
        all_eps.append(eps)

    return [np.vstack(all_bps), np.vstack(all_eps), np.vstack(annos['ignore'])]


def build_proofreading_state(state_url, datastack, render_as='url'):
    client = FrameworkClient(datastack)

    segs, annos = parse_state(state_url, datastack, client)
    sb = proofreading_statebuilder(client, segs)
    data = get_topo_points(client, segs, annos, min_size=1000)
    return sb.render_state(data,
                           return_as=render_as,
                           url_prefix=client.info.info_cache.get(datastack).get('viewer_site'))
