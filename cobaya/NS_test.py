from cobaya.run import run

run("NS_default.yaml",
    packages_path="../packages",
    output="c/noreorder",
    debug=True,
    stop_at_error=False,
    resume=True,
    force=False,
    no_mpi=True,
    test=False,
    override=None,
    proposal_mode="gamma",
    proposal_source=0
    )
