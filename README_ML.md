The components you need for a flask app:

1. app.py file
  - Has all the functions and data you need to run your app
  - Routes (through @app.route) to pages and specifies what templates to use with render_template()
  - At the end make sure you have
```  
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080, debug=True)
```

      Or else your app won't run!
      Run your app by entering "python app.py" in terminal

2. .html files in templates folder
  - These files specify
    - How your page looks
    - What data and functions are passed through your app
  - For my app,
    - index.html is just a basic home page that doesn't really do anything
    - recommender.html is the form the user fills out
    - recommendations.html does the bulk of the work with like 4 lines of code, passing the user input through my recommender function and outputting the recommendation

3. the static folder has stuff you want on your app like images, fonts, css... you can download css templates from Bootstrap to make your app look good (https://getbootstrap.com/docs/4.1/examples/)
