"""Resource Constrained Project Scheduling Problem solver"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mip

if TYPE_CHECKING:
    from typing import Iterator, Collection, Mapping

@dataclass(frozen=True)
class Resource:
    name: str
    capacity: float

@dataclass(frozen=True)
class JobData:
    name: str
    duration: int
    resource_usage: Mapping[Resource, float]
    dependencies: Collection[JobData] = ()

RESOURCES = (
    Resource('res_1', 6),
    Resource('res_2', 8),
)

# 1-indexed to keep consistent with the example diagrams
JOB_1 = JobData(
    name = 'job_1',
    duration = 3,
    resource_usage = {
        RESOURCES[0] : 5,
        RESOURCES[1] : 1,
    },
)
JOB_2 = JobData(
    name = 'job_2',
    duration = 2,
    resource_usage = {
        RESOURCES[1] : 4,
    },
)
JOB_3 = JobData(
    name = 'job_3',
    duration = 5,
    resource_usage = {
        RESOURCES[0] : 1,
        RESOURCES[1] : 4,
    },
)
JOB_4 = JobData(
    name = 'job_4',
    duration = 4,
    resource_usage = {
        RESOURCES[0] : 1,
        RESOURCES[1] : 3,
    },
    dependencies = ( JOB_1, ),
)
JOB_5 = JobData(
    name = 'job_5',
    duration = 2,
    resource_usage = {
        RESOURCES[0] : 3,
        RESOURCES[1] : 2,
    },
    dependencies = ( JOB_1, ),
)
JOB_6 = JobData(
    name = 'job_6',
    duration = 3,
    resource_usage = {
        RESOURCES[0] : 3,
        RESOURCES[1] : 1,
    },
    dependencies = ( JOB_4, ),
)
JOB_7 = JobData(
    name = 'job_7',
    duration = 4,
    resource_usage = {
        RESOURCES[0] : 2,
        RESOURCES[1] : 4,
    },
    dependencies = ( JOB_4, ),
)
JOB_8 = JobData(
    name = 'job_8',
    duration = 2,
    resource_usage = {
        RESOURCES[0] : 4,
    },
    dependencies = ( JOB_3, JOB_6, JOB_7, ),
)
JOB_9 = JobData(
    name = 'job_9',
    duration = 4,
    resource_usage = {
        RESOURCES[0] : 5,
        RESOURCES[1] : 2,
    },
    dependencies = ( JOB_2, JOB_5, JOB_6, ),
)
JOB_10 = JobData(
    name = 'job_10',
    duration = 6,
    resource_usage = {
        RESOURCES[0] : 2,
        RESOURCES[1] : 5,
    },
    dependencies = ( JOB_2, JOB_5, ),
)

JOB_CONFIG: Mapping[int, JobData] = {n : globals()[f'JOB_{n}'] for n in range(1, 11)}

PLANNING_HORIZON = sum(job.duration for job in JOB_CONFIG.values())


class Job:
    def __init__(self, model: mip.Model, data: JobData):
        self.data = data

        # binary decision variables = 1 if the job is assigned to begin at time t, 0 otherwise
        self.starts_at = [
            model.add_var(name=f'{data.name}.starts_at[{t}]', var_type=mip.BINARY)
            for t in range(PLANNING_HORIZON)
        ]
        model.add_constr( mip.xsum(self.starts_at) == 1 )

        # a linear expression that evaluates to the start time
        self.start_time = mip.xsum(t * self.starts_at[t] for t in range(PLANNING_HORIZON))
        self.completed_time = self.start_time + self.data.duration

    def possible_start_times(self, t: int) -> Iterator[int]:
        """Iterate over the possible start times that would have this job be active at time *t*."""
        return range(t, t - self.data.duration, -1)

class Event:
    def __init__(self, model: mip.Model, name: str):
        self.name = name

        self.happens_at = [
            model.add_var(name=f'{name}.start_time[{t}]', var_type=mip.BINARY)
            for t in range(PLANNING_HORIZON)
        ]
        model.add_constr( mip.xsum(self.happens_at) == 1 )

        self.time = mip.xsum(t * self.happens_at[t] for t in range(PLANNING_HORIZON))


mip_model = mip.Model()

JOBS: Mapping[int, Job] = { n : Job(mip_model, job) for n, job in JOB_CONFIG.items() }

# Constraint: resource usage
for res in RESOURCES:
    for t in range(PLANNING_HORIZON):
        total_usage = mip.xsum(
            job.data.resource_usage.get(res, 0) * job.starts_at[s]
            for job in JOBS.values()
            for s in job.possible_start_times(t) if s >= 0
        )
        mip_model.add_constr( total_usage <= res.capacity )

# Constraint: job dependencies
successors: Mapping[Job, Collection[Job]] = {
    job : [ suc for suc in JOBS.values() if job.data in suc.data.dependencies ]
    for job in JOBS.values()
}
for job, suc_list in successors.items():
    for suc in suc_list:
        mip_model.add_constr( job.completed_time <= suc.start_time )

# Objective: minimize the time at which all jobs are complete
all_done = Event(mip_model, 'all_done')
for final_job in (JOBS[8], JOBS[9], JOBS[10]):
    mip_model.add_constr( final_job.completed_time <= all_done.time )

mip_model.objective = mip.minimize( all_done.time )

if __name__ == '__main__':
    mip_model.optimize()

    print("Schedule: ")
    for n in range(1, 11):
        job = JOBS[n]
        print(f"Job {n}: begins at t={job.start_time.x} and finishes at {job.completed_time.x}")
    print("Makespan = {}".format(mip_model.objective_value))


    # sanity tests
    assert mip_model.status == mip.OptimizationStatus.OPTIMAL
    assert abs(mip_model.objective_value - 21) <= 1e-4
    mip_model.check_optimization_results()
