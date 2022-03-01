---
permalink: /blog/
title: "Blog"

excerpt: "Blog"
layout: single
author_profile: true
sitemap: true
modified: 2021-12-29
---
<h1>Latest Posts</h1>

<ul>
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
    </li>
  {% endfor %}
</ul>

<!-- ### Programming Languages
* MATLAB
* Python -->