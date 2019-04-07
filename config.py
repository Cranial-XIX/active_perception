"""
COLORS = ["gray", "red", "blue", "green"]
SHAPES = ["cube", "cylinder"]
SIZES  = ["large", "small"]
nC     = 4
nSH    = 2
nCxSH  = 8
"""

CLEVR_OBJECTS = [
    ["blue", "cube", "large"],
    ["red", "cylinder", "large"],
    ["gray", "cube", "small"],
    ["green", "cylinder", "small"]
]

sz2h   = {"large": 0.25, "small": 0.125}

IMG  = "img/"
CKPT = "ckpt/"

dim_obj   = 4
n_obj     = 4
dim_state = dim_obj * n_obj
K         = 30 # num particles

## baseline ####################################################################

b_num_layers = 1
dim_input    = 64 
dim_hidden   = 128
OE           = 10

## scene generation ############################################################

XMIN = YMIN = -2
XMAX = YMAX = 2
D_AWAY = 0.5
TALLEST = 2
