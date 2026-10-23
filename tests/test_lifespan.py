"""The app lifespan starts the indexer on startup and stops it on shutdown."""

import asyncio


class FakeIndexer:
    def __init__(self, start_blocks=False):
        self.start_blocks = start_blocks
        self.started = False
        self.start_finished = False
        self.stopped = False

    async def start(self):
        self.started = True
        if self.start_blocks:
            await asyncio.Event().wait()
        self.start_finished = True

    async def stop(self):
        self.stopped = True


def run_lifespan(server, monkeypatch, fake):
    """Build the app via main() and run its lifespan once. Returns leftover tasks."""
    monkeypatch.setattr(server, "indexer", fake)
    apps = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: apps.append(app))
    server.main()
    app = apps[0]

    async def go():
        async with app.router.lifespan_context(app):
            for _ in range(3):
                await asyncio.sleep(0)
        return asyncio.all_tasks() - {asyncio.current_task()}

    return asyncio.run(go())


def test_shutdown_stops_the_indexer(server, monkeypatch):
    fake = FakeIndexer()
    leftover = run_lifespan(server, monkeypatch, fake)
    assert fake.start_finished
    assert fake.stopped
    assert not leftover


def test_shutdown_during_initial_index_cancels_start(server, monkeypatch):
    # Otherwise start() would finish after stop() and leave a watcher running
    fake = FakeIndexer(start_blocks=True)
    leftover = run_lifespan(server, monkeypatch, fake)
    assert fake.started
    assert not fake.start_finished
    assert fake.stopped
    assert not leftover
