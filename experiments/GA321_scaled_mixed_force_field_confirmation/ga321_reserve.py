"""Run the unchanged GA321 predictor on the reserved proteins."""
from .ga321 import main


if __name__=="__main__":
    main(reserve=True)
