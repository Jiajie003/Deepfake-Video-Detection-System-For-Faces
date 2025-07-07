from new_app_cnn_lstm import db, app

with app.app_context():
    db.create_all()
    print("✅ Database created: users.db")
