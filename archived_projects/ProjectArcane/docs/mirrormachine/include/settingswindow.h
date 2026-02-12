#ifndef SETTINGSWINDOW_H
#define SETTINGSWINDOW_H

#include <QWidget>
#include <QtGui>


class SettingsWindow : public QWidget
{
  Q_OBJECT

public:

  explicit SettingsWindow(QWidget *parent = 0);

signals:

public slots:

  void closeSettings();
  void updateSettings();

private:

  QVBoxLayout *main_layout;

  QTabWidget *tab_widget;

  // general tab
  QWidget *general_tab;
  QVBoxLayout *general_layout;
  QGroupBox *model_gb;
  QGroupBox *material_gb;
  QGroupBox *misc_gb;
  QVBoxLayout *model_layout;
  QVBoxLayout *material_layout;
  QVBoxLayout *misc_layout;
  QCheckBox *cb_detail_log;
  QCheckBox *cb_create_normals;
  QCheckBox *cb_disable_bfc;
  QCheckBox *cb_mirror_v_mapping;
  QCheckBox *cb_add_collisions;
  QCheckBox *cb_generate_bsp;
  QHBoxLayout *max_leaf_size_layout;
  QLabel *max_leaf_size_label;
  QSpinBox *sb_max_leaf_size;
  QCheckBox *cb_set_blp_extension;
  QCheckBox *cb_gray_missing_textures;
  QCheckBox *cb_outdoor;
  QCheckBox *cb_indoor;
  QHBoxLayout *custom_path_layout;
  QLabel *custom_path_label;
  QLineEdit *le_custom_path;

  // WMO alpha tab
  QWidget *alpha_tab;
  QVBoxLayout *alpha_layout;
  QCheckBox *cb_portals;
  QCheckBox *cb_lights;
  QCheckBox *cb_doodads;
  QCheckBox *cb_colors;
  QCheckBox *cb_liquids;

  // the buttons down
  QHBoxLayout *buttons_layout;
  QPushButton *ok_bp;
  QPushButton *cancel_bp;

};


#endif // SETTINGSWINDOW_H
