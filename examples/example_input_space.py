from jaxzoo.spaces import Space, ContinuousSpace, DiscreteSpace, TupleSpace, DictSpace

# Define the input space
space_input = DictSpace({
    "integers" : TupleSpace([DiscreteSpace(10), DiscreteSpace(10)]),
    "embedding" : ContinuousSpace(64),
    "image" : ContinuousSpace((28, 28), low=0.0, high=1.0),
})