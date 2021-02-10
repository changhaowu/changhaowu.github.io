---
layout: page
title: Archive
permalink: /Archive/
include_nav: false
image: images/cover/C_Street2.jpeg
---

<div class="home other-pages">
  <h1 class="page-heading">Archive</h1>
  <ul class="posts">
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a> ({{ post.date | date: "%b %-d, %Y" }})
    </li>
  {% endfor %}
  </ul>
</div>