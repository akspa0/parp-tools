#include <QApplication>
#include <QtGui>

#include <window.h>


int main(int argc, char* argv[])
{
  QApplication app(argc, argv);

  Window my_window(argc, argv);
  my_window.show();

  return app.exec();
}
