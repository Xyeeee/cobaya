from cobaya.run import run

run("NS_default.yaml",
    packages_path="../packages",
    output="omg/default",
    debug=False,
    stop_at_error=False,
    resume=False,
    force=False,
    no_mpi=True,
    test=False,
    override=None,
    proposal_mode="external",
    proposal_source=1,
    beta_width=3
    )
