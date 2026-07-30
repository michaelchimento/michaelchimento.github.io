---
layout: archive
title: "First author journal publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

<p class="publication-legend">&dagger; denotes equal contribution among first authors.</p>

{% for post in site.publications reversed %}
  {% unless post.pubtype == 'other' %}
    {% include archive-single-publication.html %}
  {% endunless %}
{% endfor %}

<h2 class="publication-section">Other publications</h2>

{% for post in site.publications reversed %}
  {% if post.pubtype == 'other' %}
    {% include archive-single-publication.html %}
  {% endif %}
{% endfor %}
