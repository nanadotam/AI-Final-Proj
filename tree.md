flask_app/
│
├── static/
│   ├── css/
│   │   ├── theme.css         # Modified
│   │   └── theme.min.css     # Minified version of the modified theme.css
│   ├── js/
│   │   ├── aos.js            # Unchanged
│   │   ├── bootstrap.bundle.js  # Unchanged
│   │   ├── bootstrap.js      # Unchanged
│   │   └── custom.js         # New file for custom JS
│   └── img/
│       └── (image files like favicon.png, abstract18.webp, etc.)
│
├── templates/
│   ├── content.html
│   └── index.html            # Modified
│
├── models/
│   ├── model.h5              # Saved Keras model
│   └── model.pkl             # Alternatively, if using a pickled model
│
├── app.py                    # Modified
├── requirements.txt
└── README.md
