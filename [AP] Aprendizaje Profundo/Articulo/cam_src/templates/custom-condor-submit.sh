{% extends "condor-submit.sh" %}
{% block extra_submit_args %}
JobBatchName = "iba_ce"
#priority = 99
# RequestMemory = 4096M
#require_gpus = DeviceName == "NVIDIA RTX A5000"
#requirements = (Machine == "srvrrycarn02.priv.uco.es")
{% endblock %}

{% block extra_operation_submit_args %}
#JobBatchName = "{{ operation._jobs[0].sp.explanation_method }}"
priority = {{ 60 - operation._jobs[0].sp.partition }}
#notify_user = jbarbero@uco.es
#notification = Always
{% endblock %}
