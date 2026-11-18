# Batching heuristics for warehouse picking

This is the code repository for a follow-up paper on [Joint Order Selection, Allocation, Batching and Picking for Large Scale Warehouses](https://arxiv.org/abs/2401.04563).
Their associated code served as the starting point for our experiments and (at the time of writing) can be found at [https://github.com/zalandoresearch/batching-benchmarks].
This code repository essentially started as a fork of theirs, but has been significantly adjusted throughout the course of our work. We have kept baseline compatibility to the reference algorithms they provide.

For a discussion on our improvements and associated results, find our paper at TODO.


## How to run these experiments

This project is set up as a standard Python package with `pyproject.toml`. Use `pip` or `uv` to install this package and its dependencies.

We provide instance generator files `bin/generate_instances.py` and `bin/generate_multiphase.py` to randomly generate instances according to different size presets and write them to disk.

A solution for an instance can be computed by using the `bin/solve_instance.py` or `bin/solve_multiphase.py` file respectively.

Each of these files are set up as an easy-to-use typer CLI, so we recommend running the `--help` option to get an overview about their features and options.

As an example, see the following code, which generates a small instance for both settings and runs the DGAPQ solver on them.

```bash
python bin/generate_instances.py small -n 1
python bin/solve_instance.py dgapq data/instances/small-0

python bin/generate_multiphase.py small -n 1
python bin/solve_multiphase.py dgapq data/instances-multiphase/small-0 --n-phases 10
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
