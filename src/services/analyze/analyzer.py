import json
import os
import statistics
from pathlib import Path

def analyze(json_path: str):
    """Analyze the simulation results from a JSON file."""
    limit = int(os.getenv('RESPONSE_LIMIT', 2000))
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_agents = len(data['agent_names'])
    rounds = len(data['history'])
    
    # Средние contributions (платёж в фонд)
    avg_contributions = []
    for i in range(num_agents):
        contribs = [round_actions[i] for round_actions in data['history']]
        avg = statistics.mean(contribs)
        avg_contributions.append(avg)
    
    # Общий средний contribution
    all_contribs = [contrib for round_actions in data['history'] for contrib in round_actions]
    overall_avg_contrib = statistics.mean(all_contribs)
    
    # Финальные payoffs
    total_payoffs = data['total_payoffs']
    
    # Вывод
    print("Средний платёж для каждого агента:")
    for i, avg in enumerate(avg_contributions):
        print(f"Агент {i+1}: {avg:.2f}")
    
    print(f"Средний платёж по игре: {overall_avg_contrib:.2f}")
    
    print("Финальный результат каждого агента:")
    for i, total in enumerate(total_payoffs):
        print(f"Агент {i+1}: {total:.2f}")
    
    print(f"Суммарно: {sum(total_payoffs):.2f}")
    
    # Создать summary.json
    json_dir = Path(json_path).parent
    summary = {
        "config": data["config"],
        "agent_names": data["agent_names"],
        "history": data["history"],  # contributions per round
        "round_payoffs": data["round_payoffs"],
        "total_payoffs": data["total_payoffs"]
    }
    summary_path = json_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Создать reasoning files
    for i in range(num_agents):
        md_path = json_dir / f"reasoning_{i}.md"
        with open(md_path, 'w') as f:
            for round_detail in data['round_details']:
                r = round_detail['round']
                detail = round_detail['details'][i]
                f.write(f"## Round {r}\n\n")
                f.write("**Prompt:**\n\n")
                f.write(detail['prompt'] + "\n\n")
                f.write("**Response:**\n\n")
                response = detail['response']
                if len(response) > limit:
                    response = response[:limit] + '...'
                f.write(response + "\n\n")
