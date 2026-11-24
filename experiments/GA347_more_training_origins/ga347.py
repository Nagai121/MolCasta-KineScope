"""GA347 uses all tenfold-spaced eligible starts in the static training panel."""
from .ga318 import main


if __name__=="__main__":
    main(expanded=True, origins=10)
