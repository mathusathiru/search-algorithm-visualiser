from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
import logging
import traceback

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

FRONTEND_BUILD_DIR = Path(__file__).parent.parent / "frontend" / "build"

try:
    from algorithms.rw import generate_random_walk
    from algorithms.brw import generate_biased_random_walk
    from algorithms.bfs import generate_bfs
    from algorithms.dfs import generate_dfs
    from algorithms.dijkstra import generate_dijkstra
    from algorithms.astar import generate_astar
    from algorithms.gbfs import generate_gbfs
    from algorithms.jps import generate_jps
    from algorithms.cbs import generate_cbs
    from algorithms.icts import generate_icts
    from algorithms.mstar import generate_mstar
    from algorithms.par import generate_push_and_rotate

except ImportError as e:
    try:
        from .algorithms.rw import generate_random_walk
        from .algorithms.brw import generate_biased_random_walk
        from .algorithms.bfs import generate_bfs
        from .algorithms.dfs import generate_dfs
        from .algorithms.dijkstra import generate_dijkstra
        from .algorithms.astar import generate_astar
        from .algorithms.gbfs import generate_gbfs
        from .algorithms.jps import generate_jps
        from .algorithms.cbs import generate_cbs
        from .algorithms.icts import generate_icts
        from .algorithms.mstar import generate_mstar
        from .algorithms.par import generate_push_and_rotate
    except ImportError as e2:
        def not_implemented(*args, **kwargs):
            return [{"error": "Algorithm implementation not available", "actionType": "failed"}]
        generate_random_walk = generate_biased_random_walk = generate_bfs = generate_dfs = not_implemented
        generate_dijkstra = generate_astar = generate_gbfs = generate_jps = not_implemented
        generate_cbs = generate_icts = generate_mstar = generate_push_and_rotate = not_implemented

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/'):
        return {"error": "Invalid API endpoint"}, 404

    file_path = os.path.join(FRONTEND_BUILD_DIR, path)
    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return send_from_directory(FRONTEND_BUILD_DIR, path)

    return send_from_directory(FRONTEND_BUILD_DIR, 'index.html')

@app.route("/api/randomwalk", methods=["POST"])
def random_walk():
    data = request.get_json()

    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    if (not 0 <= start[0] < grid_size["rows"] or
        not 0 <= start[1] < grid_size["cols"] or
        not 0 <= end[0] < grid_size["rows"] or
        not 0 <= end[1] < grid_size["cols"]):
        return jsonify({
            "error": "Invalid start or end position for grid size",
            "steps": [],
            "totalSteps": 0
        }), 400

    max_steps = grid_size["rows"] * grid_size["cols"] * 2

    try:
        steps = generate_random_walk(start, end, grid_size, walls, max_steps)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in random_walk: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/biasedrandomwalk", methods=["POST"])
def biased_random_walk():
    data = request.get_json()

    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    if (not 0 <= start[0] < grid_size["rows"] or
        not 0 <= start[1] < grid_size["cols"] or
        not 0 <= end[0] < grid_size["rows"] or
        not 0 <= end[1] < grid_size["cols"]):
        return jsonify({
            "error": "Invalid start or end position for grid size",
            "steps": [],
            "totalSteps": 0
        }), 400

    max_steps = grid_size["rows"] * grid_size["cols"] * 2

    try:
        steps = generate_biased_random_walk(start, end, grid_size, walls, max_steps)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps),
            "algorithmType": "biased"
        })
    except Exception as e:
        app.logger.error(f"Error in biased_random_walk: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/bfs", methods=["POST"])
def bfs():
    try:
        data = request.get_json()

        start = tuple(data.get("start", [0, 0]))
        end = tuple(data.get("end", [0, 0]))
        grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
        walls = data.get("walls", None)

        if walls is None:
            walls = [[False for _ in range(grid_size["cols"])]
                    for _ in range(grid_size["rows"])]
        else:
            walls = [[bool(cell) for cell in row] for row in walls]

        if (not 0 <= start[0] < grid_size["rows"] or
            not 0 <= start[1] < grid_size["cols"] or
            not 0 <= end[0] < grid_size["rows"] or
            not 0 <= end[1] < grid_size["cols"]):
            app.logger.error("Invalid start or end position")
            return jsonify({"error": "Invalid start or end position for grid size", "steps": [], "totalSteps": 0}), 400

        steps = generate_bfs(start, end, grid_size, walls)

        return jsonify({"steps": steps, "totalSteps": len(steps)})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "detail": traceback.format_exc(),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/dfs", methods=["POST"])
def dfs():
    data = request.get_json()

    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    if (not 0 <= start[0] < grid_size["rows"] or
        not 0 <= start[1] < grid_size["cols"] or
        not 0 <= end[0] < grid_size["rows"] or
        not 0 <= end[1] < grid_size["cols"]):
        return jsonify({
            "error": "Invalid start or end position for grid size",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_dfs(start, end, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in dfs: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/dijkstra", methods=["POST"])
def dijkstra():
    data = request.get_json()
    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)
    initial_altitudes = data.get("initialAltitudes", None)

    try:
        steps = generate_dijkstra(start, end, grid_size, walls, initial_altitudes)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in dijkstra: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/astar", methods=["POST"])
def astar():
    data = request.get_json()
    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)
    initial_altitudes = data.get("initialAltitudes", None)

    try:
        steps = generate_astar(start, end, grid_size, walls, initial_altitudes)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in astar: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500


@app.route("/api/gbfs", methods=["POST"])
def gbfs():
    data = request.get_json()
    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)
    initial_altitudes = data.get("initialAltitudes", None)

    try:
        steps = generate_gbfs(start, end, grid_size, walls, initial_altitudes)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in gbfs: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/jps", methods=["POST"])
def jps():
    data = request.get_json()

    start = tuple(data.get("start", [0, 0]))
    end = tuple(data.get("end", [0, 0]))
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    if (not 0 <= start[0] < grid_size["rows"] or
        not 0 <= start[1] < grid_size["cols"] or
        not 0 <= end[0] < grid_size["rows"] or
        not 0 <= end[1] < grid_size["cols"]):
        return jsonify({
            "error": "Invalid start or end position for grid size",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_jps(start, end, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in jps: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/cbs", methods=["POST"])
def cbs_route():
    data = request.get_json()

    starts = [tuple(pos) for pos in data.get("starts", [[0, 0]])]
    goals = [tuple(pos) for pos in data.get("goals", [[0, 0]])]
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    for start, goal in zip(starts, goals):
        if (not 0 <= start[0] < grid_size["rows"] or
            not 0 <= start[1] < grid_size["cols"] or
            not 0 <= goal[0] < grid_size["rows"] or
            not 0 <= goal[1] < grid_size["cols"]):
            return jsonify({
                "error": "Invalid start or goal position for grid size",
                "steps": [],
                "totalSteps": 0
            }), 400

    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        return jsonify({
            "error": "Duplicate start or goal positions not allowed",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_cbs(starts, goals, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in cbs: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/icts", methods=["POST"])
def icts_route():
    data = request.get_json()

    starts = [tuple(pos) for pos in data.get("starts", [[0, 0]])]
    goals = [tuple(pos) for pos in data.get("goals", [[0, 0]])]
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    for start, goal in zip(starts, goals):
        if (not 0 <= start[0] < grid_size["rows"] or
            not 0 <= start[1] < grid_size["cols"] or
            not 0 <= goal[0] < grid_size["rows"] or
            not 0 <= goal[1] < grid_size["cols"]):
            return jsonify({
                "error": "Invalid start or goal position for grid size",
                "steps": [],
                "totalSteps": 0
            }), 400

    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        return jsonify({
            "error": "Duplicate start or goal positions not allowed",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_icts(starts, goals, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in icts: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/mstar", methods=["POST"])
def mstar_route():
    data = request.get_json()

    starts = [tuple(pos) for pos in data.get("starts", [[0, 0]])]
    goals = [tuple(pos) for pos in data.get("goals", [[0, 0]])]
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    for start, goal in zip(starts, goals):
        if (not 0 <= start[0] < grid_size["rows"] or
            not 0 <= start[1] < grid_size["cols"] or
            not 0 <= goal[0] < grid_size["rows"] or
            not 0 <= goal[1] < grid_size["cols"]):
            return jsonify({
                "error": "Invalid start or goal position for grid size",
                "steps": [],
                "totalSteps": 0
            }), 400

    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        return jsonify({
            "error": "Duplicate start or goal positions not allowed",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_mstar(starts, goals, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in mstar: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/api/pushandrotate", methods=["POST"])
def push_and_rotate():
    data = request.get_json()

    starts = [tuple(pos) for pos in data.get("starts", [[0, 0]])]
    goals = [tuple(pos) for pos in data.get("goals", [[0, 0]])]
    grid_size = data.get("gridSize", {"rows": 15, "cols": 40})
    walls = data.get("walls", None)

    if walls is None:
        walls = [[False for _ in range(grid_size["cols"])]
                for _ in range(grid_size["rows"])]
    else:
        walls = [[bool(cell) for cell in row] for row in walls]

    for start, goal in zip(starts, goals):
        if (not 0 <= start[0] < grid_size["rows"] or
            not 0 <= start[1] < grid_size["cols"] or
            not 0 <= goal[0] < grid_size["rows"] or
            not 0 <= goal[1] < grid_size["cols"]):
            return jsonify({
                "error": "Invalid start or goal position for grid size",
                "steps": [],
                "totalSteps": 0
            }), 400

    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        return jsonify({
            "error": "Duplicate start or goal positions not allowed",
            "steps": [],
            "totalSteps": 0
        }), 400

    try:
        steps = generate_push_and_rotate(starts, goals, grid_size, walls)
        return jsonify({
            "steps": steps,
            "totalSteps": len(steps)
        })
    except Exception as e:
        app.logger.error(f"Error in push_and_rotate: {str(e)}")
        return jsonify({
            "error": str(e),
            "steps": [],
            "totalSteps": 0
        }), 500

@app.route("/localhost:5000/api/<path:endpoint>", methods=["POST"])
def handle_localhost_api(endpoint):
    if endpoint == "randomwalk":
        return random_walk()
    elif endpoint == "biasedrandomwalk":
        return biased_random_walk()
    elif endpoint == "bfs":
        return bfs()
    elif endpoint == "dfs":
        return dfs()
    elif endpoint == "dijkstra":
        return dijkstra()
    elif endpoint == "astar":
        return astar()
    elif endpoint == "gbfs":
        return gbfs()
    elif endpoint == "jps":
        return jps()
    elif endpoint == "cbs":
        return cbs_route()
    elif endpoint == "icts":
        return icts_route()
    elif endpoint == "mstar":
        return mstar_route()
    elif endpoint == "pushandrotate":
        return push_and_rotate()
    else:
        return jsonify({
            "error": f"Unknown endpoint: {endpoint}",
            "steps": [],
            "totalSteps": 0
        }), 404

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "message": "Algorithm visualization API is operational"
    })

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)