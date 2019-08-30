{{ fullname | escape | underline}}

.. _{{ fullname }}:

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

-----------
Attributes
-----------

   {% block attribute %}
   {% if attributes %}
   .. autosummary::
      :toctree: {{ name }}
      :nosignatures:
      :template: autosummary/base.rst
      {% for item in attributes %}
         {%- if not item.startswith('_') %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endif %}
   {% endblock%}

--------
Methods
--------

   {% block method %}
   {% if methods %}
   .. autosummary::
      :toctree: {{ name }}
      :nosignatures:
      {% for item in methods %}
         {%- if not item.startswith('_') or item in ['__call__', '__init__'] %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endif %}
   {% endblock %}

.. rubric:: Home

* :ref:`index`

.. rubric:: Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

