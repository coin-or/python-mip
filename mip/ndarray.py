import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np

    class LinExprTensor(np.ndarray):
        """ Tensor of :class:`~mip.Var` or :class:`~mip.LinExpr` elements

        This is a Numpy ndarray subclass with the only purpose to change the default
        behaviour when the comparison operators are used.
        For the operators :code:`<=`, :code:`>=`, :code:`==`, the default ndarray 
        behavior is to perform element-wise comparisons and then cast the results to
        :code:`bool`. This class does not cast the results to :code:`bool`.

        It is otherwise a completely normal Numpy ndarray, on which to operate with
        Numpy functions and methods for any linear algebra purpose, and can be used in
        operations with other :code:`int` or :code:`float` ndarrays.

        Even though it is possible to create an instance of LinExprTensor on your own,
        it is seldom necessary: you can use :class:`~mip.Model` method :code:`add_var_tensor()`
        to create a LinExprTensor containing :class:`~mip.Var` elements, and perform
        linear operations on that to build your model.
        """

        def __new__(cls, *args, **kwargs):
            obj = super(LinExprTensor, cls).__new__(*args, **kwargs)
            return obj

        def __array_finalize__(self, obj):
            # ``self`` is a new object resulting from
            # ndarray.__new__(LinExprTensor, ...), therefore it only has
            # attributes that the ndarray.__new__ constructor gave it -
            # i.e. those of a standard ndarray.
            #
            # We could have got to the ndarray.__new__ call in 3 ways:
            # From an explicit constructor - e.g. LinExprTensor():
            #    obj is None
            #    (we're in the middle of the LinExprTensor.__new__
            #    constructor, and self.info will be set when we return to
            #    LinExprTensor.__new__)
            if obj is None:
                return
            # From view casting - e.g arr.view(InfoArray):
            #    obj is arr
            #    (type(obj) can be LinExprTensor)
            # From new-from-template - e.g infoarr[:3]
            #    type(obj) is LinExprTensor
            #
            # Note that it is here, rather than in the __new__ method,
            # that we set the default value for fields, because this
            # method sees all creation of default objects - with the
            # LinExprTensor.__new__ constructor, but also with
            # arr.view(LinExprTensor).
            #
            # example:
            # self.field = getattr(obj, 'field', None)
            #
            # We do not need to return anything

        def __le__(self, other):
            return np.less_equal(self, other, dtype=object)

        def __ge__(self, other):
            return np.greater_equal(self, other, dtype=object)

        def __eq__(self, other):
            return np.equal(self, other, dtype=object)


except ImportError:
    logger.debug("Unable to import numpy", exc_info=True)

    class LinExprTensor:
        pass
