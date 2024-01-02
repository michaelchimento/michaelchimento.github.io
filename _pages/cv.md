---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.A. in Politics, New York University, 2012
* M.Sc. in Evolution of Language and Cognition, 2018
* Ph.D in Biology, IMPRS for Organismal Biology / Konstanz University, 2022

Skills
======
* Experimental design
* Data analysis
* Programming
  * Python (advanced)
  * R (advanced)
* Writing

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>

Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
