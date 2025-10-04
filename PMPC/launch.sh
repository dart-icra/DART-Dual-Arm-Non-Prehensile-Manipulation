#!/bin/bash

# DualArmNPT Launch Script
# Provides easy commands for common experiment types

print_usage() {
    echo "Usage: ./launch.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  simple     - Run basic control (main.py)"
    echo "  parallel   - Run parallel MPC (main_parallel.py)" 
    echo "  enhanced   - Run full experiment (main_parallel_enhanced.py)"
    echo "  demo       - Run demo with predefined targets"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh simple --object_name cube"
    echo "  ./launch.sh parallel --target 0.1 0 0.05 0 0 0 --runtime 30"
    echo "  ./launch.sh enhanced --object_name sphere --mass 1.5"
    echo "  ./launch.sh demo"
}

run_demo() {
    echo "Running demo experiments..."
    echo "1. Cube positioning..."
    python main_parallel_enhanced.py --target 0.08 0 0.05 0 0 0 --object_name cube --runtime 25 --tolerance 0.005
    
    echo ""
    echo "2. Cylinder manipulation..." 
    python main_parallel_enhanced.py --target 0.06 0 0.08 0 0 0 --object_name cylinder --runtime 30 --mass 1.0
    
    echo ""
    echo "3. Sphere control..."
    python main_parallel_enhanced.py --target 0.10 0 -0.06 0 0 0 --object_name sphere --runtime 35 --friction 0.3
}

# Check if command provided
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

COMMAND=$1
shift  # Remove command from arguments

case $COMMAND in
    simple)
        echo "Running simple control..."
        python main.py "$@"
        ;;
    parallel)
        echo "Running parallel MPC control..."
        python main_parallel.py "$@"
        ;;
    enhanced)
        echo "Running enhanced experiment..."
        python main_parallel_enhanced.py "$@"
        ;;
    demo)
        run_demo
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        print_usage
        exit 1
        ;;
esac