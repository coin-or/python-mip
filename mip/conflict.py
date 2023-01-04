import logging
from enum import Enum

import mip
import mip.constants

logger = logging.getLogger("conflict")


class IISFinderAlgorithm(Enum):
    DELETION_FILTER = 1
    ADDITIVE_ALGORITHM = 2


class ConflictFinder:
    """This class groups some IIS (Irreducible Infeasible Set) search algorithms"""

    def __init__(self, model: mip.Model):
        if model.status == mip.OptimizationStatus.LOADED:
            logger.debug("model not runned yet, checking if feasible or not")
            model.emphasis = 1  # feasibility
            model.preprocess = 1  # -1  automatic, 0  off, 1  on.
            model.optimize()
        assert (
            model.status == mip.OptimizationStatus.INFEASIBLE
        ), "model is not linear infeasible"
        self.model = model

    def find_iis(
        self, method: IISFinderAlgorithm = IISFinderAlgorithm.DELETION_FILTER
    ) -> mip.ConstrList:
        """main method to find an IIS, this method is just a grouping of the other implementations

        Args:
            model (mip.Model): Infeasible model where to find the IIS
            method (str, optional): name of the method to use ["deletion-filter", "additive_algorithm"]. Defaults to 'deletion-filter".

        Returns:
            mip.ConstrList: IIS constraint list
        """
        # assert ,is not because time limit
        if method == IISFinderAlgorithm.DELETION_FILTER:
            return self.deletion_filter()
        if method == IISFinderAlgorithm.ADDITIVE_ALGORITHM:
            return self.additive_algorithm()

    def deletion_filter(self) -> mip.ConstrList:
        """deletion filter algorithm for search an IIS

        Args:
            model (mip.Model): Infeasible model

        Returns:
            mip.ConstrList: IIS
        """
        # 1. create a model with all constraints but one
        aux_model = self.model.copy()
        aux_model.objective = 1
        aux_model.emphasis = 1  # feasibility
        aux_model.preprocess = 1  # -1  automatic, 0  off, 1  on.

        logger.debug("starting deletion_filter algorithm")

        for inc_crt in self.model.constrs:
            aux_model_inc_crt = aux_model.constr_by_name(
                inc_crt.name
            )  # find constraint by name
            aux_model.remove(aux_model_inc_crt)  # temporally remove inc_crt

            aux_model.optimize()
            status = aux_model.status
            # 2. test feasibility, if feasible, return dropped constraint to the set
            # 2.1 else removed it permanently
            # logger.debug('status {}'.format(status))
            if status == mip.OptimizationStatus.INFEASIBLE:
                logger.debug("removing permanently {}".format(inc_crt.name))
                continue
            elif status in [
                mip.OptimizationStatus.FEASIBLE,
                mip.OptimizationStatus.OPTIMAL,
            ]:
                aux_model.add_constr(
                    inc_crt.expr, name=inc_crt.name, priority=inc_crt.priority
                )

        iis = aux_model.constrs

        return iis

    def additive_algorithm(self) -> mip.ConstrList:
        """Additive algorithm to find an IIS

        Returns:
            mip.ConstrList: IIS
        """
        # Create some aux models to test feasibility of the set of constraints
        aux_model_testing = mip.Model()
        for var in self.model.vars:
            aux_model_testing.add_var(
                name=var.name,
                lb=var.lb,
                ub=var.ub,
                var_type=var.var_type,
                # obj= var.obj,
                # column=var.column   #!! libc++abi.dylib: terminating with uncaught exception of type CoinError
            )
        aux_model_testing.objective = 1
        aux_model_testing.emphasis = 1  # feasibility
        aux_model_testing.preprocess = 1  # -1  automatic, 0  off, 1  on.
        aux_model_iis = (
            aux_model_testing.copy()
        )  # a second aux model to test feasibility of the incumbent iis

        # algorithm start
        all_constraints = self.model.constrs
        testing_crt_set = mip.ConstrList(model=aux_model_testing)  # T
        iis = mip.ConstrList(model=aux_model_iis)  # I

        while True:
            for crt in all_constraints:
                testing_crt_set.add(crt.expr, name=crt.name)
                aux_model_testing.constrs = testing_crt_set
                aux_model_testing.optimize()

                if aux_model_testing.status == mip.OptimizationStatus.INFEASIBLE:
                    iis.add(crt.expr, name=crt.name)
                    aux_model_iis.constrs = iis
                    aux_model_iis.optimize()

                    if aux_model_iis.status == mip.OptimizationStatus.INFEASIBLE:
                        return iis
                    elif aux_model_iis.status in [
                        mip.OptimizationStatus.FEASIBLE,
                        mip.OptimizationStatus.OPTIMAL,
                    ]:
                        testing_crt_set = mip.ConstrList(model=aux_model_testing)
                        for (
                            crt
                        ) in (
                            iis
                        ):  # basically this loop is for set T=I // aux_model_iis =  iis.copy()
                            testing_crt_set.add(crt.expr, name=crt.name)
                        break

    def deletion_filter_milp_ir_lc_bd(self) -> mip.ConstrList:
        """Integer deletion filter algorithm (milp_ir_lc_bd)

        Raises:
            NotImplementedError: [description]

        Returns:
            mip.ConstrList: [description]
        """
        raise NotImplementedError("WIP")
        # major constraint sets definition
        t_aux_model = mip.Model(name="t_auxiliary_model")
        iis_aux_model = mip.Model(name="t_auxiliary_model")

        linear_constraints = mip.ConstrList(
            model=t_aux_model
        )  # all the linear model constraints
        variable_bound_constraints = mip.ConstrList(
            model=t_aux_model
        )  # all the linear model constrants related specifically for the variable bounds
        integer_varlist_crt = mip.VarList(
            model=t_aux_model
        )  # the nature vars constraints for vartype in Integer/Binary

        # fill the above sets with the constraints
        for crt in self.model.constrs:
            linear_constraints.add(crt.expr, name=crt.name)
        for var in self.model.vars:
            if var.lb != -mip.INF:
                variable_bound_constraints.add(
                    var >= var.lb, name="{}_lb_crt".format(var.name)
                )
            if var.ub != mip.INF:
                variable_bound_constraints.add(
                    var <= var.ub, name="{}_ub_crt".format(var.name)
                )
        for var in self.model.vars:
            if var.var_type in (mip.INTEGER, mip.BINARY):
                integer_varlist_crt.add(var)

        status = "IIS"
        # add all LC,BD to the incumbent, T= LC + BD
        for (
            var
        ) in (
            self.model.vars
        ):  # add all variables as if they where CONTINUOUS and without bonds (because this will be separated)
            iis_aux_model.add_var(
                name=var.name, lb=-mip.INF, ub=mip.INF, var_type=mip.CONTINUOUS
            )
        for crt in linear_constraints + variable_bound_constraints:
            iis_aux_model.add_constr(crt.expr, name=crt.name, priority=crt.priority)

        iis_aux_model.optimize()
        if iis_aux_model.status == mip.OptimizationStatus.INFEASIBLE:
            # if infeasible means that this is a particular version of an LP
            return self.deletion_filter()  # (STEP 2)

        # add all the integer constraints to the model
        iis_aux_model.vars.remove(
            [var for var in integer_varlist_crt]
        )  # remove all integer variables
        for var in integer_varlist_crt:
            iis_aux_model.add_var(
                name=var.name,
                lb=-mip.INF,
                ub=mip.INF,
                var_type=var.var_type,  # this will add the var with his original type
            )
        # filter IR constraints that create infeasibility (STEP 1)
        for var in integer_varlist_crt:
            iis_aux_model.vars.remove(iis_aux_model.var_by_name(var.name))
            iis_aux_model.add_var(
                name=var.name,
                lb=-mip.INF,
                ub=mip.INF,
                var_type=mip.CONTINUOUS,  # relax the integer constraint over var
            )
            iis_aux_model.optimize()
            # if infeasible then update incumbent T = T-{ir_var_crt}
            # else continue
        # STEP 2 filter lc constraints
        # STEP 3 filter BD constraints
        # return IS o IIS

    def deletion_filter_milp_lc_ir_bd(self) -> mip.ConstrList:
        # TODO
        raise NotImplementedError


class ConflictRelaxer:
    def __init__(self, model: mip.Model):
        if model.status == mip.OptimizationStatus.LOADED:
            logger.debug("model not runned yet, checking if feasible or not")
            model.emphasis = 1  # feasibility
            model.preprocess = 1  # -1  automatic, 0  off, 1  on.
            model.optimize()
        assert (
            model.status == mip.OptimizationStatus.INFEASIBLE
        ), "model is not linear infeasible"

        self.model = model
        self.iis_num_iterations = 0
        self.iis_iterations = []
        self.relax_slack_iterations = []

    @property
    def slack_by_crt(self) -> dict:
        answ = {}
        for slack_dict_iter in self.relax_slack_iterations:
            for crt_name in slack_dict_iter.keys():
                if crt_name in answ.keys():
                    answ[crt_name] += slack_dict_iter[crt_name]
                else:
                    answ[crt_name] = slack_dict_iter[crt_name]
        return answ

    def hierarchy_relaxer(
        self,
        relaxer_objective: str = "min_abs_slack_val",
        default_priority: mip.constants.ConstraintPriority = mip.constants.ConstraintPriority.MANDATORY,
    ) -> mip.Model:
        """hierarchy relaxer algorithm, it's gonna find a IIS and then relax it using the objective function defined (`relaxer_objective`) and then update the model
        with the relaxed constraints. This process runs until there's not more IIS on the model.

        Args:
            relaxer_objective (str, optional): objective function of the relaxer model (IIS relaxer model). Defaults to 'min_abs_slack_val'.
            default_priority (ConstraintPriority, optional): If a constraint does not have a supported substring priority in the name, it will assign a default priority.
                                                             Defaults to ConstraintPriority.MANDATORY.

        Raises:
            Exception: [description]

        Returns:
            mip.Model: relaxed model
        """

        relaxed_model = self.model.copy()
        relaxed_model._status = self.model._status  # TODO solve this in a different way

        # map unmaped constraitns to default
        for crt in relaxed_model.constrs:
            if not crt.priority:
                crt.priority = default_priority

        cf = ConflictFinder(relaxed_model)
        while True:
            # 1. find iis
            iis = cf.find_iis(IISFinderAlgorithm.DELETION_FILTER)
            self.iis_iterations.append([crt.name for crt in iis])  # track iteration
            self.iis_num_iterations += 1  # track iteration

            iis_priority_list = [crt.priority for crt in iis]
            # check if "relaxable" model mapping
            if set(iis_priority_list) == set(
                [mip.constants.ConstraintPriority.MANDATORY]
            ):
                raise Exception(
                    "Infeasible model, is not possible to relax MANDATORY constraints"
                )

            # 2. relax iis
            slack_dict = self.relax_iis(iis, relaxer_objective=relaxer_objective)

            self.relax_slack_iterations.append(slack_dict)
            # 3. add the slack variables to the original problem
            relaxed_model = self.relax_constraints(relaxed_model, slack_dict)

            # 4. check if feasible
            relaxed_model.emphasis = 1  # feasibility
            relaxed_model.optimize()
            if relaxed_model.status in [
                mip.OptimizationStatus.FEASIBLE,
                mip.OptimizationStatus.OPTIMAL,
            ]:
                logger.debug("finished relaxation process !")
                break
            else:
                logger.debug(
                    "relaxed the current IIS, still infeasible, searching for a new IIS to relax"
                )
                logger.debug("relaxed constraints {0}".format(list(slack_dict.keys())))

        return relaxed_model

    @classmethod
    def relax_iis(
        cls, iis: mip.ConstrList, relaxer_objective: str = "min_abs_slack_val"
    ) -> dict:

        """This function is the sub module that finds the optimum relaxation for an IIS, given a crt priority mapping and a objective function

        Args:
            iis (mip.ConstrList): IIS constraint list
            relaxer_objective (str, optional): objective function to use when relaxing. Defaults to 'min_abs_slack_val'.

        Returns:
            dict: a slack variable dictionary with the value of the {constraint_name:slack.value} pair to be added to each constraint in order to make the IIS feasible
        """
        relax_iis_model = mip.Model()
        lowest_priority = min([crt.priority for crt in iis])
        to_relax_crts = [crt for crt in iis if crt.priority == lowest_priority]

        # create a model that only contains the iis
        slack_vars = {}
        abs_slack_vars = {}
        abs_slack_cod_vars = {}
        for crt in iis:
            for var in crt._Constr__model.vars:
                relax_iis_model.add_var(
                    name=var.name,
                    lb=var.lb,
                    ub=var.ub,
                    var_type=var.var_type,
                    obj=var.obj,
                )
            if crt in to_relax_crts:
                # if this is a -toberelax- constraint
                slack_vars[crt.name] = relax_iis_model.add_var(
                    name="{0}__{1}".format(crt.name, "slack"),
                    lb=-mip.INF,
                    ub=mip.INF,
                    var_type=mip.CONTINUOUS,
                )

                abs_slack_vars[crt.name] = relax_iis_model.add_var(
                    name="{0}_abs".format(slack_vars[crt.name].name),
                    lb=0,
                    ub=mip.INF,
                    var_type=mip.CONTINUOUS,
                )

                # add relaxed constraint to model
                relax_expr = crt.expr + slack_vars[crt.name]
                relax_iis_model.add_constr(
                    relax_expr,
                    name="{}_relaxed".format(crt.name),
                )

                # add abs(slack) variable encoding constraints
                relax_iis_model.add_constr(
                    abs_slack_vars[crt.name] >= slack_vars[crt.name],
                    name="{}_positive_min_bound".format(slack_vars[crt.name].name),
                )
                relax_iis_model.add_constr(
                    abs_slack_vars[crt.name] >= -slack_vars[crt.name],
                    name="{}_negative_min_bound".format(slack_vars[crt.name].name),
                )

            else:
                # if not to be relaxed we added directly to the model
                relax_iis_model.add_constr(
                    crt.expr, name="{}_original".format(crt.name), priority=crt.priority
                )

        # find the min abs value of the slack variables
        relax_iis_model.objective = mip.xsum(list(abs_slack_vars.values()))
        relax_iis_model.sense = mip.MINIMIZE
        relax_iis_model.optimize()
        if relax_iis_model.status == mip.OptimizationStatus.INFEASIBLE:
            raise ValueError(
                "sub relaxation model infeasible, this could mean that in the IIS the mandatory constraints are infeasible sometimes. Also could mean that the big_m parameter is overflowed "
            )

        slack_dict = {}
        for crt in to_relax_crts:
            slack_dict[crt.name] = slack_vars[crt.name].x

        return slack_dict

    @classmethod
    def relax_constraints(cls, relaxed_model: mip.Model, slack_dict: dict) -> mip.Model:
        """this method creates a modification of the model `relaxed_model` where all the constraints in the slack_dict are
        modified in order to add the slack values to make the IIS disappear

        Args:
            relaxed_model (mip.Model): model to relax
            slack_dict (dict): pairs {constraint_name: slack_var.value}

        Returns:
            mip.Model: a modification of the original model where all the constraints are modified with the slack values
        """
        for crt_name in slack_dict.keys():
            crt_original = relaxed_model.constr_by_name(crt_name)

            relax_expr = crt_original.expr + slack_dict[crt_name]
            relaxed_model.add_constr(
                relax_expr, name=crt_original.name, priority=crt_original.priority
            )
            relaxed_model.remove(crt_original)  # remove constraint

        return relaxed_model
