from flask import Flask, request, jsonify
from OR_tool_prototype import recommend_agents, DEFAULT_MAX_GRACE_PERIOD
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
        
        # Get optional max_grace_period parameter
        max_grace_period = data.get('max_grace_period', DEFAULT_MAX_GRACE_PERIOD)
        
        # Call the recommendation service
        recommendations = recommend_agents(new_task, agents, current_tasks, max_grace_period)
        
        # Parse the JSON string back to a dict
        recommendations_dict = json.loads(recommendations)
        
        return jsonify(recommendations_dict), 200
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Use 0.0.0.0 to listen on all available network interfaces
    app.run(host='0.0.0.0', port=port) 