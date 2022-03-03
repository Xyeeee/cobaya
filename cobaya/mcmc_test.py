from cobaya.run import run

run("camb_default.yaml",
    packages_path="packages",
    output="runs/camb_default",
    debug=False,
    stop_at_error=True,
    resume=False,
    force=False,
    no_mpi=False,
    test=False,
    override=None
    )
