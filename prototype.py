import itertools
from datetime import datetime, timedelta
from typing import List, Tuple
import requests

# --- CONFIG ---
OSRM_URL = "https://osrm-caribbean-785077267034.us-central1.run.app/route/v1/driving/"
cache = {}

# --- CLASSES ---

class Task:
    def __init__(self, pickup_id: str, delivery_id: str,
                 pickup_coords: Tuple[float, float],
                 delivery_coords: Tuple[float, float],
                 ready_time: datetime,
                 delivery_duration: timedelta):
        self.pickup_id = pickup_id
        self.delivery_id = delivery_id
        self.pickup_coords = pickup_coords
        self.delivery_coords = delivery_coords
        self.ready_time = ready_time
        self.delivery_duration = delivery_duration

class Stop:
    def __init__(self, task: Task, is_pickup: bool):
        self.task = task
        self.is_pickup = is_pickup

    def location(self):
        return self.task.pickup_coords if self.is_pickup else self.task.delivery_coords

    def id(self):
        return f"{'P' if self.is_pickup else 'D'}-{self.task.pickup_id}"

class Agent:
    def __init__(self, agent_id: str, current_location: Tuple[float, float]):
        self.agent_id = agent_id
        self.schedule: List[Task] = []
        self.current_location = current_location

# --- UTILITIES ---

def get_travel_time(start: Tuple[float, float], end: Tuple[float, float]) -> timedelta:
    if start == end:
        return timedelta(seconds=0)
    key = (start, end)
    if key in cache:
        return cache[key]
    coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    response = requests.get(OSRM_URL + coordinates, params={"overview": "false"})
    data = response.json()
    if "routes" in data and data["routes"]:
        seconds = data["routes"][0]["duration"]
        duration = timedelta(seconds=seconds)
        cache[key] = duration
        return duration
    else:
        raise ValueError("Invalid OSRM response or unreachable route.")

def generate_valid_sequences(tasks: List[Task]) -> List[List[Stop]]:
    pickups = [Stop(task, True) for task in tasks]
    deliveries = [Stop(task, False) for task in tasks]

    all_stops = pickups + deliveries
    valid_sequences = []

    for perm in itertools.permutations(all_stops):
        seen_pickups = set()
        valid = True
        for stop in perm:
            if stop.is_pickup:
                seen_pickups.add(stop.task.pickup_id)
            else:
                if stop.task.pickup_id not in seen_pickups:
                    valid = False
                    break
        if valid:
            valid_sequences.append(list(perm))
    return valid_sequences

def simulate_sequence_with_batching(start_location: Tuple[float, float], stops: List[Stop]) -> Tuple[int, timedelta, List[Tuple[str, datetime, bool]]]:
    current_time = datetime.now()
    location = start_location
    results = []
    late_count = 0

    # Group stops by (location, is_pickup) to simulate batching
    grouped_stops = []
    seen = set()
    for stop in stops:
        key = (stop.location(), stop.is_pickup)
        if key not in seen:
            group = [s for s in stops if s.location() == stop.location() and s.is_pickup == stop.is_pickup]
            grouped_stops.append(group)
            seen.add(key)

    for group in grouped_stops:
        stop = group[0]
        travel_time = get_travel_time(location, stop.location()) if location != stop.location() else timedelta(seconds=0)
        arrival_time = current_time + travel_time

        for s in group:
            if s.is_pickup:
                wait_time = max(s.task.ready_time - arrival_time, timedelta(seconds=0))
                arrival_time += wait_time
            else:
                due_time = s.task.ready_time + s.task.delivery_duration
                is_late = arrival_time > due_time
                if is_late:
                    late_count += 1
                results.append((s.id(), arrival_time, is_late))

        current_time = arrival_time
        location = stop.location()

    total_duration = current_time - datetime.now()
    return late_count, total_duration, results

def rank_agents_for_task(agents: List[Agent], new_task: Task):
    agent_scores = []

    for agent in agents:
        all_tasks = agent.schedule + [new_task]
        sequences = generate_valid_sequences(all_tasks)
        best_late = float('inf')
        best_duration = None
        best_seq = None
        best_results = None
        for seq in sequences:
            late_count, total_duration, results = simulate_sequence_with_batching(agent.current_location, seq)
            if late_count < best_late or (late_count == best_late and (best_duration is None or total_duration < best_duration)):
                best_late = late_count
                best_duration = total_duration
                best_seq = seq
                best_results = results
        if best_seq is not None:
            score = 1000 - (best_late * 100) - int(best_duration.total_seconds() / 60)
            explanation = generate_agent_report(agent, new_task, best_seq, best_results, best_late)
            agent_scores.append({
                "agent_id": agent.agent_id,
                "score": score,
                "late_deliveries": best_late,
                "total_duration_minutes": int(best_duration.total_seconds() / 60),
                "explanation": explanation
            })

    agent_scores.sort(key=lambda x: (-x["score"], x["late_deliveries"], x["total_duration_minutes"]))
    return agent_scores[:3]

def generate_agent_report(agent: Agent, task: Task, sequence: List[Stop], results: List[Tuple[str, datetime, bool]], late_count: int) -> str:
    report_lines = [f"Agent {agent.agent_id} schedule with new task {task.pickup_id}:"]
    for stop_id, arrival, late in results:
        status = "LATE" if late else "on time"
        report_lines.append(f"  Stop {stop_id} arrives at {arrival.strftime('%H:%M:%S')} ({status})")
    if late_count == 0:
        report_lines.append("All deliveries on time.")
    else:
        report_lines.append(f"{late_count} deliveries will be late.")
    return "\n".join(report_lines)

# --- SAMPLE USAGE ---

if __name__ == "__main__":
    # Define agents
    agent1 = Agent("agent_1", current_location=(17.1189, -61.8406))
    agent2 = Agent("agent_2", current_location=(17.1500, -61.8500))
    agent3 = Agent("agent_3", current_location=(17.1250, -61.8600))  # starts with no tasks

    # Existing tasks for agent1 (4 tasks)
    agent1.schedule.extend([
        Task("A1", "A1", (17.1175, -61.8456), (17.1335, -61.8315), datetime.now() + timedelta(minutes=2), timedelta(minutes=20)),
        Task("A2", "A2", (17.1180, -61.8460), (17.1340, -61.8300), datetime.now() + timedelta(minutes=3), timedelta(minutes=22)),
        Task("A3", "A3", (17.1190, -61.8470), (17.1350, -61.8290), datetime.now() + timedelta(minutes=4), timedelta(minutes=19)),
        Task("A4", "A4", (17.1200, -61.8480), (17.1360, -61.8280), datetime.now() + timedelta(minutes=5), timedelta(minutes=23))
    ])

    # Existing tasks for agent2 (3 tasks)
    agent2.schedule.extend([
        Task("B1", "B1", (17.1210, -61.8490), (17.1370, -61.8270), datetime.now() + timedelta(minutes=3), timedelta(minutes=25)),
        Task("B2", "B2", (17.1220, -61.8500), (17.1380, -61.8260), datetime.now() + timedelta(minutes=4), timedelta(minutes=20)),
        Task("B3", "B3", (17.1230, -61.8510), (17.1390, -61.8250), datetime.now() + timedelta(minutes=6), timedelta(minutes=22))
    ])

    agents = [agent1, agent2, agent3]

    # Single new task for testing
    new_task = Task(
        "ORDER123",
        "ORDER123",
        (17.1240, -61.8450),
        (17.1400, -61.8240),
        datetime.now() + timedelta(minutes=5),
        timedelta(minutes=20)
    )

    ranked_agents = rank_agents_for_task(agents, new_task)

    output = {
        "task_id": new_task.pickup_id,
        "suggestions": []
    }

    for agent_info in ranked_agents:
        output["suggestions"].append({
            "agent_id": agent_info["agent_id"],
            "score": agent_info["score"],
            "late_deliveries": agent_info["late_deliveries"],
            "total_duration_minutes": agent_info["total_duration_minutes"],
            "explanation": agent_info["explanation"]
        })

    import json
    print(json.dumps(output, indent=2))