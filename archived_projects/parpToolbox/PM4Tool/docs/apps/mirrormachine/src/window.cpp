#include <string>

#include <settingswindow.h>
#include <model.h>
#include <WMO_exporter.h>

#include <window.h>


Window::Window(int argc, char* argv[], QWidget *parent)
    : QWidget(parent)
{
  QSettings opt("egamh9", "mirrormachine");

  _layout = new QGridLayout(this);

  _filler = new QLabel();
  _filler->setPixmap(QPixmap(":img/mm.jpg"));

  _open_file_button = new QPushButton("Open &File...");
  QObject::connect(_open_file_button,
                   SIGNAL(clicked()),
                   this,
                   SLOT(openFile()));
  _path_label = new QLabel();
  _path_label->setWordWrap(true);
  _path_label->setAlignment(Qt::AlignCenter);
  if (argc >= 1)
    _path_label->setText(QString(argv[1]));

  // imports
  _import_cbox = new QComboBox();
  _import_cbox->addItem("3DS file (.3ds)");
  _import_cbox->addItem("OBJ file (.obj)");
  _import_cbox->addItem("WMO/v14 file (.wmo)");
  _import_cbox->setCurrentIndex(opt.value("import_cbox", 0).toInt());
  QObject::connect(_import_cbox,
                   SIGNAL(currentIndexChanged(int)),
                   this,
                   SLOT(saveCurrentImportIndex(int)));
  // exports
  _export_cbox = new QComboBox();
  _export_cbox->addItem("WMO v17 (.wmo)");

  _opt_button = new QPushButton("S&ettings");
  QObject::connect(_opt_button,
                   SIGNAL(clicked()),
                   this,
                   SLOT(openSettings()));
  _start_button = new QPushButton("&Start convertion");
  QObject::connect(_start_button,
                   SIGNAL(clicked()),
                   this,
                   SLOT(startConversion()));
  _about_button = new QPushButton("About");
  QObject::connect(_about_button,
                   SIGNAL(clicked()),
                   this,
                   SLOT(aboutMM()));
  _exit_button = new QPushButton("E&xit");
  QObject::connect(_exit_button,
                   SIGNAL(clicked()),
                   this,
                   SLOT(close()));

  _layout->addWidget(_filler, 0, 0, 1, 4);
  _layout->addWidget(_open_file_button, 1, 0, 1, 1);
  _layout->addWidget(_path_label, 1, 1, 1, 3);
  _layout->addWidget(_import_cbox, 2, 0, 1, 2);
  _layout->addWidget(_export_cbox, 2, 2, 1, 2);
  _layout->addWidget(_opt_button, 3, 0);
  _layout->addWidget(_start_button, 3, 1);
  _layout->addWidget(_about_button, 3, 2);
  _layout->addWidget(_exit_button, 3, 3);
  setLayout(_layout);

  setWindowTitle("MirrorMachine - 1.2a - Unexpected sex edition");
}


void Window::openSettings()
{
  SettingsWindow *sw = new SettingsWindow();
  sw->show();
}
void Window::openFile()
{
  _path_label->setText(
      QFileDialog::getOpenFileName(this, "Select a file to import"));
}

void Window::saveCurrentImportIndex(int index)
{
  QSettings opt("egamh9", "mirrormachine");
  opt.setValue("import_cbox", index);
}

void Window::aboutMM()
{
  QMessageBox aboutBox;
  aboutBox.setIconPixmap(QPixmap(":img/about.jpg"));
  aboutBox.setWindowTitle("About MirrorMachine");
  aboutBox.setText("<p>MirrorMachine is developed by ShgCk (aka Gamh), "
                   "for friends from the Architecture Department "
                   "and the Modcraft people.</p>"
                   "<p>A neat homepage can be found "
                   "<a href=\"http://egamh9.net/dev/mm/index.php\">here</a>, "
                   "and from there you can contact me if you want to.</p>"
                   "<p>Coded for learning/fun during winter 2012 "
                   "and beyond. Have fun using this tool !</p>");
  aboutBox.setStyleSheet("p { text-align: center; }");
  aboutBox.exec();
}





void Window::startConversion()
{
  if (!_path_label->text().isEmpty())
  {
    int result = 0;
    std::string output_path = _path_label->text().toStdString();
    cleanOutputPath(&output_path);

    // start model import
    Model imp_model(_path_label->text().toStdString().c_str(),
                    _import_cbox->currentIndex());

    //*
    // if it succeed then converts it
    if ((imp_model._import_flag & Model::FATAL_ERROR) != Model::FATAL_ERROR)
    {
      // start convertion for selected file format
      switch (_export_cbox->currentIndex())
      {
        case 0: // WMO
          WMO_exporter exp_wmo;
          result = exp_wmo.process(imp_model, output_path.c_str());
          break;
      }
      // display a global success/fail message
      switch (result)
      {
        case 0: // ok
          QMessageBox::information(this,
                                   "Success",
                                   "Your file has been "
                                   "successfully converted.");
          break;
        default: // misc fail
          QMessageBox::warning(this,
                               "Fail",
                               "Failed to write the files, "
                               "probably can't access drive.");
          break;
      }
      // display specific warnings
      if (imp_model._import_flag & Model::MISSING_UV)
        QMessageBox::warning(this,
                             "Warning",
                             "The import file hasn't all "
                             "its mapping coordinates !");
      if (imp_model._import_flag & Model::OUTRANGE_UV)
        QMessageBox::warning(this,
                             "Warning",
                             "Some of your mapping coordinates "
                             "are out of the [0, 1] range !");
    }
    // else import failed and we stop there
    else
    {
      if (imp_model._import_flag & Model::BAD_OBJ_ORDER)
        QMessageBox::critical(this,
                              "Error",
                              "This OBJ file isn't well formed. "
                              "Read MirrorMachine home page for more info.");
      else
        QMessageBox::critical(this,
                              "Error",
                              "The import failed, probably wrong format.");
    }
    //*/
  }
  else
  {
    QMessageBox::warning(this,
                         "Failed",
                         "You didn't specify any file to import");
  }

}

void Window::cleanOutputPath(std::string *path)
{
  std::string ext = path->substr(path->length() - 4, path->length());
  if ((ext.compare(".3ds") == 0) ||
      (ext.compare(".3DS") == 0) ||
      (ext.compare(".obj") == 0) ||
      (ext.compare(".OBJ") == 0))
  {
    *path = path->substr(0, path->length() - 4);
  }
}


