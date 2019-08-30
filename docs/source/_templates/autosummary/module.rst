{{ fullname | escape | underline }}

.. _{{ fullname }}:

.. 
    .. currentmodule:: {{ name }}


.. 
    .. rubric:: Description

Description
************

.. automodule:: {{ module }}.{{ name }}



{% if functions %}
..   
    .. rubric:: Functions

Functions
**********

.. autosummary:: 
    :toctree: {{ name }}
    :nosignatures:
    {% for function in functions %}
        {{ function }}
    {% endfor %}

{% endif %}


{% if classes %}
..   
    .. rubric:: Classes

Classes
*********

.. autosummary::
    :toctree: {{ name }}
    :nosignatures:
    {% for class in classes %}
        {{ class }}
    {% endfor %}

{% endif %}

.. rubric:: Home

* :ref:`index`

.. rubric:: Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



