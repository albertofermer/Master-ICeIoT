{% extends "base_script.sh" %}
{% block body %}

{% for operation in operations %}
# {{ operation.cmd }}
{% set job_id = environment.aggregate_id_fn(operation._jobs) %}
{% set output_folder = "./condor_output/" + job_id + "/" + operation.name %}
mkdir -p "{{ output_folder }}"
{% endfor %}


condor_submit <<SUBMITEND
universe = vanilla
transfer_executable = False
getenv = True
{% block extra_submit_args %}
{% endblock %}

{% for operation in operations %}
{% set executable = operation.cmd.split()[1] %}
{% set arguments = ' '.join(operation.cmd.split()[2:]) %}
{% set job_id = environment.aggregate_id_fn(operation._jobs) %}
{% set project_id = project %}
{% set output_folder = "./condor_output/" + job_id + "/" + operation.name %}
+signacProjectId = "{{ project_id }}"
+signacJobId = "{{ job_id }}"
+signacOperationId = "{{ operation.id }}"
+signacOperationName = "{{ operation.name }}"
Executable = {{ executable }}
Arguments  = {{ arguments }}
request_cpus = {{ operation.directives.np }}
request_gpus = {{ operation.directives.ngpu }}

{% set memory_requested = operation.directives.memory  %}
{% if memory_requested %}
request_memory = {{ memory_requested }}
{% endif %}

{% block extra_operation_submit_args scoped %}
{% endblock %}

Log        = {{ output_folder }}/log
Output     = {{ output_folder }}/stdout
Error      = {{ output_folder }}/stderr
Queue

{% endfor %}
SUBMITEND
{% endblock %}
