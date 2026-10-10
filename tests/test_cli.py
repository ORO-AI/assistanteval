from assistanteval import cli


def test_each_command_reaches_its_module(capsys):
    assert cli.main([]) == 2
    assert (
        "usage: assistanteval {run,grade,report,annotate,serve,harbor-export,harbor-verify}"
        in capsys.readouterr().err
    )
    for command in cli.COMMANDS:
        try:
            cli.main([command, "--help"])
        except SystemExit as stop:
            assert stop.code == 0
        assert "usage:" in capsys.readouterr().out
