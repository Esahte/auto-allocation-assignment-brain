"""Shared region-policy matching for fleet pre-filtering and solver checks."""

from typing import Any, Dict, Iterable, List, Optional, Tuple


VALID_SHARING_MODES = {'open_access', 'region_first'}
VALID_ROUTE_COVERAGE = {'entire_route', 'any_endpoint'}


def _legacy_defaults(name: str) -> Tuple[str, str]:
    """Missing-field defaults used only while dashboard/OR-Tools deploys are staggered."""
    if 'scooter' in str(name or '').lower():
        return 'open_access', 'entire_route'
    return 'region_first', 'any_endpoint'


def normalize_region(region: Any) -> Dict[str, Any]:
    """Normalize dashboard, FleetState, or GeofenceRegion data into one contract."""
    if hasattr(region, 'region_id'):
        raw = {
            'id': getattr(region, 'region_id'),
            'name': getattr(region, 'region_name', ''),
            'polygon': getattr(region, 'polygon', []),
            'agent_ids': getattr(region, 'fleet_ids', []),
            'sharing_mode': getattr(region, 'sharing_mode', None),
            'route_coverage': getattr(region, 'route_coverage', None),
        }
    else:
        raw = region or {}

    region_id = str(raw.get('id', raw.get('region_id', '')))
    name = str(raw.get('name', raw.get('region_name', '')))
    legacy_sharing, legacy_coverage = _legacy_defaults(name)
    sharing_mode = raw.get('sharing_mode')
    route_coverage = raw.get('route_coverage')
    if sharing_mode is None:
        sharing_mode = legacy_sharing
    if route_coverage is None:
        route_coverage = legacy_coverage

    if sharing_mode not in VALID_SHARING_MODES:
        raise ValueError(f"Invalid sharing_mode for region {region_id}: {sharing_mode}")
    if route_coverage not in VALID_ROUTE_COVERAGE:
        raise ValueError(f"Invalid route_coverage for region {region_id}: {route_coverage}")

    polygon = []
    for point in raw.get('polygon', []):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            polygon.append((float(point[0]), float(point[1])))

    agent_source = raw.get('agent_ids', raw.get('fleet_ids', []))
    return {
        'id': region_id,
        'name': name,
        'polygon': polygon,
        'agent_ids': {str(agent_id) for agent_id in agent_source},
        'sharing_mode': sharing_mode,
        'route_coverage': route_coverage,
    }


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    lat, lng = point
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        lat_i, lng_i = polygon[current]
        lat_j, lng_j = polygon[previous]
        if (lat_i > lat) != (lat_j > lat):
            edge_lng = lng_i + ((lat - lat_i) * (lng_j - lng_i) / (lat_j - lat_i))
            if lng < edge_lng:
                inside = not inside
        previous = current
    return inside


def _location_tuple(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if hasattr(value, 'lat') and hasattr(value, 'lng'):
        return float(value.lat), float(value.lng)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def active_task_endpoints(task: Any) -> List[Tuple[float, float]]:
    pickup = _location_tuple(getattr(task, 'restaurant_location', None))
    delivery = _location_tuple(getattr(task, 'delivery_location', None))
    job_type = str(getattr(task, 'job_type', 'PAIRED') or 'PAIRED').upper()
    if job_type in {'DELIVERY', 'DELIVERY_ONLY'}:
        return [delivery] if delivery is not None else []
    if job_type in {'PICKUP', 'PICKUP_ONLY'}:
        return [pickup] if pickup is not None else []
    return [endpoint for endpoint in (pickup, delivery) if endpoint is not None]


def task_matches_region(task: Any, region: Any) -> bool:
    normalized = normalize_region(region)
    endpoints = active_task_endpoints(task)
    if not endpoints or len(normalized['polygon']) < 3:
        return False
    endpoint_matches = [
        point_in_polygon(endpoint, normalized['polygon'])
        for endpoint in endpoints
    ]
    if normalized['route_coverage'] == 'entire_route':
        return all(endpoint_matches)
    return any(endpoint_matches)


def matching_regions(task: Any, regions: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized_regions = [normalize_region(region) for region in regions]
    return [region for region in normalized_regions if task_matches_region(task, region)]


def evaluate_region_eligibility(
    agent_id: str,
    task: Any,
    regions: Iterable[Any],
    online_agent_ids: Iterable[str],
) -> Dict[str, Any]:
    """Apply overlap union, Region-first precedence, fallback, and reverse locks."""
    normalized_agent_id = str(agent_id)
    normalized_regions = [normalize_region(region) for region in regions]
    online_ids = {str(value) for value in online_agent_ids}
    matched = [region for region in normalized_regions if task_matches_region(task, region)]
    matched_ids = {region['id'] for region in matched}
    agent_regions = [
        region for region in normalized_regions
        if normalized_agent_id in region['agent_ids']
    ]
    staffed_region_first = [
        region for region in matched
        if region['sharing_mode'] == 'region_first'
        and bool(region['agent_ids'] & online_ids)
    ]
    permitted_member_ids = set().union(
        *(region['agent_ids'] for region in matched)
    ) if matched else set()
    agent_matches_subscription = any(
        region['id'] in matched_ids for region in agent_regions
    )
    distance_bypass = any(
        normalized_agent_id in region['agent_ids']
        for region in staffed_region_first
    )

    result = {
        'eligible': True,
        'reason': None,
        'matching_regions': matched,
        'matching_region_ids': [region['id'] for region in matched],
        'agent_region_ids': [region['id'] for region in agent_regions],
        'staffed_region_first_ids': [region['id'] for region in staffed_region_first],
        'permitted_member_ids': sorted(permitted_member_ids),
        'restrict_to_members': bool(staffed_region_first),
        'distance_bypass': distance_bypass,
    }

    if agent_regions and not agent_matches_subscription:
        result.update(eligible=False, reason='agent_restricted_to_region')
    elif staffed_region_first and normalized_agent_id not in permitted_member_ids:
        result.update(eligible=False, reason='not_in_task_region')
    return result
