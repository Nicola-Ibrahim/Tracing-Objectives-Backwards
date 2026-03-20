import sys
import os

# This script should be run from the 'backend' directory as:
# python3 -m src.verify_di (if moved to src) 
# or just python3 verify_di.py if we fix the paths.

# Let's fix paths for running from project root
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.startup import BackendStartUp

def verify():
    print("Verifying DI Container...")
    try:
        config = {"redis_url": "redis://localhost:6379/0"}
        backend = BackendStartUp(config)
        backend.initialize_modules()
        
        container = backend.container
        
        # Test resolutions
        dispatcher = container.evaluation.task_dispatcher()
        print(f"✓ TaskDispatcher resolved: {type(dispatcher)}")
        
        run_service = container.evaluation.run_diagnostics_service()
        print(f"✓ RunDiagnosticsService resolved: {type(run_service)}")
        
        modeling_service = container.modeling.service()
        print(f"✓ ModelingService resolved: {type(modeling_service)}")
        
        print("\nDI Verification SUCCESSFUL!")
        
    except Exception as e:
        print(f"\nDI Verification FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
