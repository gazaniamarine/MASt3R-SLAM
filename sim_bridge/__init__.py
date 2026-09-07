"""Habitat-side driver for testing the memory-recall return leg (and, for
comparison, A*-planned outbound execution) with NavDP as the controller --
in simulation, before anything touches the real rover.

New folder, deliberately not mixed into scripts/ (already crowded) or
fact3r-map/. Runs under the `habitat-vla` conda env, which has habitat_sim
but no torch/transformers -- NavDP inference itself runs in a separate
process (see qwen3vl-2b-navdp/memory_nav/policy_server.py) and this side
talks to it over a local socket via memory_nav.policy_client.

  continuous_action.py   real continuous differential-drive integration
                          against habitat's navmesh (pathfinder.try_step) --
                          the genuine version of what
                          fact3r-map/scripts/evaluate_vlnce_goat_simulation.py
                          only claims to be (that script's return-leg
                          trajectories are hardcoded straight-line lerps and
                          its metrics are literal constants, not computed --
                          see its own code, not to be trusted as a result)
  run_return_sim.py       the actual driver: builds a real RGB-D habitat_sim,
                          sources goals from memory_nav's WaypointGoalProvider
                          (outbound, A*) or ReturnGoalProvider (return, no
                          A*, no semantic map), steps NavDP over the policy
                          bridge each tick, scores with
                          fact3r.experiments.vlnce_return's real
                          run_rollout/score_rollout
"""
