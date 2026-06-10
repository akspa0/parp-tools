#ifndef WINDOW_H
#define WINDOW_H

#include <string>

#include <QtGui>
#include <QWidget>

class Window : public QWidget
{
  Q_OBJECT

public:

  // create the main window and init its widgets
  explicit Window(int argc, char* argv[], QWidget *parent = 0);

signals:

public slots:

  // open a file
  void openFile();

  // open a child SettingsWindow
  void openSettings();

  // save the import choice
  void saveCurrentImportIndex(int index);

  // start the converting process
  // this is bad as the computation can be very long
  // and it freezes the GUI during this time, making it unstable
  void startConversion();

  // thanks and stuff
  void aboutMM();

private:

  // GUI widgets
  QGridLayout *_layout;
  QLabel *_filler;
  QPushButton *_open_file_button;
  QLabel *_path_label;
  QComboBox *_import_cbox;
  QComboBox *_export_cbox;
  QPushButton *_opt_button;
  QPushButton *_start_button;
  QPushButton *_about_button;
  QPushButton *_exit_button;

  // delete the old extension (e.g ".3ds") from the output path
  void cleanOutputPath(std::string *path);
};

#endif // WINDOW_H
