from cobaya.run import run

run("NS_default.yaml",
    packages_path="../packages",
    output="run/NS_default",
    debug=True,
    stop_at_error=False,
    resume=False,
    force=False,
    no_mpi=True,
    test=False,
    override=None
    )
