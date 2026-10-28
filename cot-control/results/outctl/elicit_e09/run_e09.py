"""elicit.py at Inkling thinking effort 0.9, written under elicit_e09/runs/ (elicit.py unchanged).  From cot-control/rl:  python ../results/outctl/elicit_e09/run_e09.py MODEL VARIANTS SPLIT N"""
import asyncio, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../elicit"))
import elicit  # loads the Astra items from its own directory
elicit.O, elicit.TML_EFFORT = HERE, 0.9
sys.argv = ["elicit.py"] + sys.argv[1:]
asyncio.run(elicit.main())
