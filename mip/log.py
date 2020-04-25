class ProgressLog:
    """Class to store the improvement of lower
    and upper bounds over time during the search.
    Results stored here are useful to analyze the
    performance of a given formulation/parameter setting
    for solving a instance. To be able to automatically
    generate summarized experimental results, fill the
    :attr:`~mip.model.ProgressLog.instance` and
    :attr:`~mip.model.ProgressLog.settings` of this object with the instance
    name and formulation/parameter setting details, respectively.

    Attributes:
        log(Tuple[float, Tuple[float, float]]): Tuple in the format :math:`(time, (lb, ub))`, where :math:`time` is the processing time and :math:`lb` and :math:`ub` are the lower and upper bounds, respectively

        instance(str): instance name

        settings(str): identification of the formulation/parameter
        settings used in the optimization (whatever is relevant to
        identify a given computational experiment)
    """

    def __init__(self):
        self.log = []

        self.instance = ""

        self.settings = ""

    def write(self, file_name: str = ""):
        """Saves the progress log. If no extension is informed,
        the :code:`.plog` extension will be used. If only a directory is
        informed then the name will be built considering the
        :attr:`~mip.model.ProgressLog.instance` and
        :attr:`~mip.model.ProgressLog.settings` attributes"""
        if not self.instance:
            raise ValueError(
                "Enter model name (instance name) to save experimental data."
            )
        if not file_name:
            file_name = "{}-{}.plog".format(self.instance, self.settings)
        else:
            if file_name.endswith("/") or file_name.endswith("\\"):
                file_name += "{}-{}.plog".format(self.instance, self.settings)

        if not file_name.endswith(".plog"):
            file_name += ".plog"

        f = open(file_name, "w")
        f.write("instance: {}".format(self.instance))
        f.write("settings: {}".format(self.settings))
        for (s, (l, b)) in self.log:
            f.write("{},{},{}".format(s, l, b))
        f.close()

    def read(self, file_name: str):
        """Reads a progress log stored in a file"""
        f = open(file_name, "r")
        lin = f.next()
        self.instance = lin.split(":")[1].lstrip()
        self.settings = lin.split(":")[1].lstrip()
        for lin in f:
            cols = lin.split(",")
            (s, (l, b)) = (float(cols[0]), (float(cols[1]), float(cols[2])))
            self.log.append((s, (l, b)))
        f.close()
