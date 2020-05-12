---
layout: page
title: "Home"
class: home
---

# Hi, I'm Arnab! 

<div class="columns" markdown="1">

<div class="intro" markdown="1">
I am a Ph.D. candidate at [NanoEnergy & Thermophysics lab](https://netlab.umasscreate.net/) in the Department of Electrical and Computer Engineering, University of Massachusetts Amherst. My Ph.D. advisor is [Prof. Zlatan Aksamija](https://blogs.umass.edu/zlatana/about/). 

<div>During my stay UMass Amherst, I have developed scientific computational models based on first-principles Density Functional Theory calculations for understanding charge and heat flow in two-dimensional (2D) van der Waals (vdW) semiconductors, including graphene, transition metal dichalcogenides and beyond-graphene materials, and in their heterostructures. Beyond developing electro-thermal simulations for 2D nanoelectronic devices, my research interests also encompass developing data-accelerated multiphysics device models to improve our understanding of complex material systems.</div>

</div>

<div class="me" markdown="1">
<picture>
  <source srcset='/images/me.webp' type='image/webp' />
  <img
    src='/images/me.jpeg'
    alt='Arnab Majee'/>
</picture>

{:.no-list}
* <a href="mailto:{{ site.email }}">{{ site.email }}</a>
</div>

</div>

<!---My favorite pastimes are eating & debating. I also like to read & travel quite a bit. (whenever I get the chance). -->

## Featured Publications

<div class="featured-publications">
  {% for pub in site.data.publications %}
    {% if pub.highlight %}
      <a href="{{ pub.pdf }}" class="publication">
        <strong>{{ pub.title }}</strong>
        <span class="authors">{% for author in pub.authors %}{{ author }}{% unless forloop.last %}, {% endunless %}{% endfor %}</span>.
        <i>{{ pub.venue }}, {{ pub.vol }}, {{ pub.year }}, {{ pub.pages}}</i>.
        {% for award in pub.awards %}<br/><span class="award"><i class="fas fa-{% if award == "Best Paper Award" %}trophy{% else %}award{% endif %}" aria-hidden="true"></i> {{ award }}</span>{% endfor %}
      </a>
    {% endif %}
  {% endfor %}
</div>

<a href="{{ "/publications/" | relative_url }}" class="button">
  <i class="fas fa-chevron-circle-right"></i>
  Show All Publications
</a>

## Featured Projects

<div class="featured-projects">
  {% assign sorted_projects = site.data.projects | sort: 'highlight' %}
  {% for project in sorted_projects %}
    {% if project.highlight %}
      {% include project.html project=project %}
    {% endif %}
  {% endfor %}
</div>
<a href="{{ "/projects/" | relative_url }}" class="button">
  <i class="fas fa-chevron-circle-right"></i>
  Show More Projects
</a>


<div class="news-travel" markdown="1">

<div class="news" markdown="1">
## News

<ul>
{% for news in site.data.news limit:10 %}
  {% include news.html news=news %}
{% endfor %}
</ul>

</div>

<!-- 
<div class="travel" markdown="1">
## Travel

<table>
<tbody>
{% assign future_travel = site.data.travel | where_exp:'item','item.start == null' %}
{% for travel in future_travel %}
  {% include travel.html travel=travel %}
{% endfor %}
{% assign sorted_travel = site.data.travel | where_exp:'item','item.start' | sort: 'start' | reverse %}
{% for travel in sorted_travel limit:12 %}
  {% include travel.html travel=travel %}
{% endfor %}
</tbody>
</table>

</div>
-->

</div>
