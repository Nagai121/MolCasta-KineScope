"""Train the expanded data pool after all GA392 generation lanes finish."""
from .ga386 import main
if __name__=="__main__":
    main(expanded_pool=True)
