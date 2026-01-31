


from jinja2 import Template

coordinates = [
    {'lat': 37.7749, 'lon': -122.4194},
    {'lat': 37.3382, 'lon': -121.8863},
    {'lat': 34.0522, 'lon': -118.2437},
    # add more coordinates here
]

with open('map.html') as f:
    template = Template(f.read())

html = template.render(coordinates=coordinates)

with open('map_output.html', 'w') as f:
    f.write(html)
