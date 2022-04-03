from cobaya.run import run

run("NS_default.yaml",
    packages_path="../packages",
    output="h/default",
    debug=True,
    stop_at_error=False,
    resume=False,
    force=False,
    no_mpi=True,
    test=False,
    override=None,
    proposal_mode="beta",
    proposal_source=0,
    beta_width=3
    )
