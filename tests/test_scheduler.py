from ecofl.federated.scheduler import EnergyAwareScheduler, SchedulerConfig


class DummyClient:
    def __init__(self, client_id, cpu=10.0, ram=10.0, energy=100.0):
        self.client_id = client_id
        self._status = {
            "cpu_percent": cpu,
            "ram_percent": ram,
            "energy_budget_remaining_mj": energy,
        }

    def get_resource_status(self):
        return self._status


def test_scheduler_selects_eligible_clients():
    scheduler = EnergyAwareScheduler(SchedulerConfig(min_clients_per_round=1))
    clients = [
        DummyClient(0, cpu=10, ram=10, energy=100),
        DummyClient(1, cpu=95, ram=10, energy=100),
    ]
    eligible, excluded = scheduler.select_clients(clients, current_round=1)
    assert [c.client_id for c in eligible] == [0]
    assert excluded == [1]


def test_scheduler_early_termination_after_patience():
    scheduler = EnergyAwareScheduler(SchedulerConfig(convergence_tolerance=0.01, patience=2, max_rounds=20))
    assert scheduler.should_terminate(0.80, 1) is False
    assert scheduler.should_terminate(0.805, 2) is False
    assert scheduler.should_terminate(0.806, 3) is True
