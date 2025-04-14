import getpass
import subprocess
import errno
from typing import List

from flow import get_aggregate_id
from flow.environment import ComputeEnvironment
from flow.errors import SubmitError
from flow.scheduling.base import Scheduler, JobStatus, ClusterJob


def _fetch_condor_job_status(user=None):
    """Fetch the cluster jobs status information from the Condor scheduler."""
    def parse_status(s: int):
        codes = {
            0: JobStatus.submitted,
            1: JobStatus.queued,
            2: JobStatus.active,
            3: JobStatus.error,
            # 4 "completed"
            5: JobStatus.held,
            6: JobStatus.error,
        }
        return codes.get(s, JobStatus.registered)

    if user is None:
        user = getpass.getuser()

    cmd = ['condor_q', '-submitter', user, '-nobatch', '-autoformat', 'signacOperationId', 'JobStatus']
    try:
        result = subprocess.check_output(cmd).decode('utf-8', errors='backslashreplace')
    except subprocess.CalledProcessError:
        raise
    except IOError as error:
        if error.errno != errno.ENOENT:
            raise
        else:
            raise RuntimeError("Condor not available.")

    lines = result.split('\n')
    for line in lines:
        if line.strip():
            fields = line.split()
            job_id = fields[0]
            job_status = int(fields[1])

            yield ClusterJob(job_id, parse_status(job_status))


class CondorScheduler(Scheduler):
    submit_cmd = ['bash']

    def __init__(self, user=None, **kwargs):
        super(CondorScheduler, self).__init__(**kwargs)
        self.user = user

    def jobs(self):
        """Yield cluster jobs by querying the scheduler."""
        self._prevent_dos()
        yield from _fetch_condor_job_status(user=self.user)

    def submit(self, script: str, pretend: bool = False, flags: List[str] = None, **kwargs):
        """Submit a job script for execution to the scheduler.

        :param script:
            The job script submitted for execution.
        :type script:
            str
        :param pretend:
            If True, do not actually submit the script, but only simulate the submission.
            Can be used to test whether the submission would be successful.
            Please note: A successful "pretend" submission is not guaranteed to succeed.
        :type pretend:
            bool
        :param flags:
            Additional arguments to pass through to the scheduler submission command.
        :type flags:
            list
        :returns:
            Returns True if the cluster job was successfully submitted, otherwise None.
        """
        submit_cmd = self.submit_cmd

        if pretend:
            print("# Submit command: {}".format('  '.join(submit_cmd)))
            print(script)
            print()
        else:
            try:
                p = subprocess.Popen(submit_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
                p.stdin.write(script.encode('utf-8'))
                p.stdin.close()
            except subprocess.CalledProcessError as e:
                raise SubmitError("condor_submit error: {}".format(e.output))

            return True

    @classmethod
    def is_present(cls):
        """Return True if it appears that a Condor scheduler is available within the environment."""
        try:
            subprocess.check_output(['condor_q', '-version'], stderr=subprocess.STDOUT)
        except (IOError, OSError):
            return False
        else:
            return True


class CondorEnvironment(ComputeEnvironment):
    template = 'condor-submit.sh'
    scheduler_type = CondorScheduler
    aggregate_id_fn = get_aggregate_id