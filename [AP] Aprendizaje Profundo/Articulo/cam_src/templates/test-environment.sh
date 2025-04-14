{% extends "base_script.sh" %}
{% block body %}
--> {{ project }}
{% for operation in operations %}
{% set executable = operation.cmd.split()[0] %}
{% set arguments = ' '.join(operation.cmd.split()[1:]) %}
{% set job_id = operation.cmd.split()[-1] %}
{% set project_id = project %}
{{ executable }}
{{ arguments }}
{{ operation.name }}
{% endfor %}
{% endblock %}
