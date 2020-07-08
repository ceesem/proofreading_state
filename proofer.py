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
                              grouped=False,
                              bp_tags=BP_TAGS,
                              bp_color=BP_COLOR,
                              ep_tags=EP_TAGS,
                              ep_color=EP_COLOR,
                              stop_color=STOP_COLOR):
    img = nglui.statebuilder.ImageLayerConfig(
        source=client.info.image_source(), contrast_controls=True)
    seg = nglui.statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(), fixed_ids=fixed_ids)

    if grouped:
        bp_points = nglui.statebuilder.PointMapper(
            point_column='pt', group_column='group', set_position=True)
    else:
        bp_points = nglui.statebuilder.PointMapper(
            point_column='pt', set_position=True)
    bp_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='branch_points', color=bp_color, mapping_rules=bp_points, tags=bp_tags, active=True)
    sb_bp = nglui.statebuilder.StateBuilder([img, seg, bp_annos])

    ep_points = nglui.statebuilder.PointMapper()
    ep_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='end_points', color=ep_color, mapping_rules=ep_points, array_data=True, tags=ep_tags, active=False)
    sb_ep = nglui.statebuilder.StateBuilder([ep_annos])

    stop_points = nglui.statebuilder.SphereMapper()
    stop_annos = nglui.statebuilder.AnnotationLayerConfig(
        name='axon_point', color=stop_color, mapping_rules=stop_points, array_data=True, active=False)
    sb_stop = nglui.statebuilder.StateBuilder([stop_annos])
    return nglui.statebuilder.ChainedStateBuilder([sb_bp, sb_ep, sb_stop])


def _branch_and_end_points_ordered(nrn):
    cps = nrn.skeleton.cover_paths
    cp_ends = np.array([nrn.skeleton.parent_nodes(cp[-1]) for cp in cps])

    bps = nrn.skeleton.branch_points
    cp_queue = [0]
    primary_cps = [ii for ii, cp in enumerate(cp_ends) if cp == nrn.skeleton.root]
    cp_queue.extend(primary_cps)

    is_done = np.full(len(cp_ends), False)
    bp_groups = []
    eps = []
    while not np.all(is_done):
        ind_to_do = cp_queue.pop(0)
        is_done[ind_to_do] = True
        cp = cps[ind_to_do]

        # Get branch points
        bps_on_cp = bps[np.isin(bps, cp)]
        bp_groups.append(bps_on_cp[bps_on_cp != nrn.skeleton.root])

        # Get end point
        eps.append(cp[0])

        # Get other cps to check
        cp_offshoots = np.flatnonzero(np.isin(cp_ends, cp))
        if len(cp_offshoots) > 0:
            cp_offshoots = cp_offshoots[~is_done[cp_offshoots]]
            lens = np.array([nrn.skeleton.distance_to_root[cps[ii][0]] -
                             nrn.skeleton.distance_to_root[cps[ii][-1]] for ii in cp_offshoots])
            cp_offshoots = cp_offshoots[np.argsort(lens)]
            for ind in cp_offshoots:
                if ind not in cp_queue:
                    cp_queue.insert(0, ind)

    bp_points = []
    for skinds in bp_groups:
        if len(skinds) > 0:
            bp_points.append(nrn.skeleton.vertices[skinds])

    ep_points = []
    for skind in eps:
        ep_points.append(nrn.skeleton.vertices[skind])
    ep_points = np.vstack(ep_points)

    return bp_points, ep_points

# def _end_points_ordered(nrn):
#     eps = nrn.end_points
#     ep_ord = np.argsort(nrn.distance_to_root(eps))
#     return nrn.mesh.vertices[eps[ep_ord]]


def get_topo_points(client, segs, annos, min_size=1000, voxel_resolution=np.array([4, 4, 40])):
    mm = trimesh_io.MeshMeta(cv_path=client.info.segmentation_source(),
                             map_gs_to_https=True, disk_cache_path='meshes')

    multi_seg = len(segs) > 1

    all_bps = []
    all_eps = []
    for ii, oid in enumerate(segs):
        mesh = mm.mesh(seg_id=oid)

        lcs = mesh_filters.filter_largest_component(mesh)
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

        bps, eps = _branch_and_end_points_ordered(nrn)
        all_bps.append([bp/voxel_resolution for bp in bps])
        all_eps.append(eps/voxel_resolution)
        # bps = _branch_points_ordered(nrn) / voxel_resolution
        # all_bps.append(bps)

        # eps = _end_points_ordered(nrn) / voxel_resolution
        # all_eps.append(eps)

    return [all_bps, np.vstack(all_eps), np.vstack(annos['ignore'])]


def _format_data(data, sphere_radius=1000):
    all_bps, all_eps, ignore_points = data
    bp_groups = []
    ii = 0
    all_bps_flat = []
    for bpg in all_bps:
        for bps in bpg:
            all_bps_flat.append(bps)
            bp_groups.append(np.full(len(bps), ii))
            ii += 1

    all_bps = np.vstack(all_bps_flat)
    bp_groups = np.concatenate(bp_groups)
    bp_df = pd.DataFrame(data={'pt': all_bps.tolist(), 'group': bp_groups})
    ignore_points = (ignore_points, np.full(len(ignore_points), sphere_radius))
    return bp_df, all_eps, ignore_points


def build_proofreading_state(state_url, datastack, render_as='url'):
    client = FrameworkClient(datastack)

    segs, annos = parse_state(state_url, datastack, client)
    sb = proofreading_statebuilder(client, segs)
    data = get_topo_points(client, segs, annos, min_size=1000)
    data = _format_data(data)
    return sb.render_state(data,
                           return_as=render_as,
                           url_prefix=client.info.info_cache.get(datastack).get('viewer_site'))
