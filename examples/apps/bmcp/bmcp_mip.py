from mip.model import Model, xsum
from mip.constants import MINIMIZE, BINARY
from bmcp_data import BMCPData
from typing import List
from itertools import product

def build_mip(data: BMCPData,
              U: List[int]  # list of available channels
              ) -> Model:
    return m
