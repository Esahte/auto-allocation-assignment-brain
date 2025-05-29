from flask import Flask, request, jsonify
from OR_tool_prototype import recommend_agents
import json
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['new_task', 'agents', 'current_tasks']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Get the data
        new_task = data['new_task']
        agents = data['agents']
        current_tasks = data['current_tasks']
        
        # Call the recommendation service
        recommendations = recommend_agents(new_task, agents, current_tasks)
        
        # Parse the JSON string back to a dict
        recommendations_dict = json.loads(recommendations)
        
        return jsonify(recommendations_dict), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8081
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port) 