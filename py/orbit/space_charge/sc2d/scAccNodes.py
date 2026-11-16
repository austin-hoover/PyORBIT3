from orbit.space_charge.scAccNodes import SC_Base_AccNode


class SC2D_UniformEllipses_AccNode(SC_Base_AccNode):
    def __init__(self, sc_calculator, name="no name"):
        super().__init__(sc_calculator, name)
        self.setType("UnifEllsSC2D")

    def track(self, paramsDict):
        if not self.switcher:
            return
        self.sc_calculator.trackBunch(paramsDict["bunch"], self.sc_length)
