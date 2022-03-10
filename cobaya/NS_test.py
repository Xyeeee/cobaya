from cobaya.run import run

run("NS_default.yaml",
    packages_path="../packages",
    output="runs/NS_default",
    debug=True,
    stop_at_error=True,
    resume=False,
    force=False,
    no_mpi=True,
    test=False,
    override=None
    )
