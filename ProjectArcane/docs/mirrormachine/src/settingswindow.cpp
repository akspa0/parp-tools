#include <settingswindow.h>

SettingsWindow::SettingsWindow(QWidget *parent)
    : QWidget(parent)
{
  main_layout = new QVBoxLayout;


  tab_widget = new QTabWidget;

  // GENERAL TAB
  // misc converting parameters
  general_tab = new QWidget;
  general_layout = new QVBoxLayout;

  model_gb = new QGroupBox("Model settings");
  model_layout = new QVBoxLayout;
  cb_add_collisions = new QCheckBox("Add collisions");
  cb_generate_bsp = new QCheckBox("Generate true BSP tree");
  QObject::connect(cb_generate_bsp, SIGNAL(toggled(bool)),
                   cb_add_collisions, SLOT(setDisabled(bool)));
  max_leaf_size_layout = new QHBoxLayout();
  max_leaf_size_label = new QLabel("Set max leaf size");
  sb_max_leaf_size = new QSpinBox();
  max_leaf_size_layout->addWidget(max_leaf_size_label);
  max_leaf_size_layout->addWidget(sb_max_leaf_size);
  sb_max_leaf_size->setRange(0, 10000);
  cb_create_normals = new QCheckBox("Create vertex normals from face normals (slow)");
  custom_path_layout = new QHBoxLayout;
  custom_path_label = new QLabel("Custom file path");
  le_custom_path = new QLineEdit;
  custom_path_layout->addWidget(custom_path_label);
  custom_path_layout->addWidget(le_custom_path);
  cb_set_blp_extension= new QCheckBox("Set BLP extension to textures");
  cb_indoor = new QCheckBox("Set indoor definitions");
  cb_outdoor = new QCheckBox("Set outdoor definitions");
  model_layout->addWidget(cb_add_collisions);
  model_layout->addWidget(cb_generate_bsp);
  model_layout->addLayout(max_leaf_size_layout);
  model_layout->addWidget(cb_create_normals);
  model_layout->addLayout(custom_path_layout);
  model_layout->addWidget(cb_set_blp_extension);
  model_layout->addWidget(cb_indoor);
  model_layout->addWidget(cb_outdoor);
  model_gb->setLayout(model_layout);

  material_gb = new QGroupBox("Materials");
  material_layout = new QVBoxLayout;
  cb_mirror_v_mapping = new QCheckBox("Mirror V texture mapping");
  cb_disable_bfc = new QCheckBox("Disable backface culling");
  cb_gray_missing_textures= new QCheckBox("Set gray missing textures");
  material_layout->addWidget(cb_mirror_v_mapping);
  material_layout->addWidget(cb_disable_bfc);
  material_layout->addWidget(cb_gray_missing_textures);
  material_gb->setLayout(material_layout);

  misc_gb = new QGroupBox("Misc settings");
  misc_layout = new QVBoxLayout;
  cb_detail_log = new QCheckBox("Verbose log (will output everything)");
  misc_layout->addWidget(cb_detail_log);
  misc_gb->setLayout(misc_layout);

  general_layout->addWidget(model_gb);
  general_layout->addWidget(material_gb);
  general_layout->addWidget(misc_gb);
  general_tab->setLayout(general_layout);

  // ALPHA TAB
  // specific options for alpha
  alpha_tab = new QWidget;
  alpha_layout = new QVBoxLayout;

  cb_portals = new QCheckBox("Import portals");
  cb_lights = new QCheckBox("Import lights");
  cb_doodads = new QCheckBox("Import doodads");
  cb_colors = new QCheckBox("Import colors (vertex shading)");
  cb_liquids = new QCheckBox("Import liquids");

  alpha_layout->addWidget(cb_portals);
  alpha_layout->addWidget(cb_lights);
  alpha_layout->addWidget(cb_doodads);
  alpha_layout->addWidget(cb_colors);
  alpha_layout->addWidget(cb_liquids);
  alpha_tab->setLayout(alpha_layout);

  tab_widget->addTab(general_tab, "General options");
  tab_widget->addTab(alpha_tab, "WMO/v14");


  // BUTTONS
  buttons_layout = new QHBoxLayout;
  ok_bp = new QPushButton("OK");
  QObject::connect(ok_bp, SIGNAL(clicked()), this, SLOT(updateSettings()));
  cancel_bp = new QPushButton("Cancel");
  QObject::connect(cancel_bp, SIGNAL(clicked()), this, SLOT(closeSettings()));
  buttons_layout->addWidget(ok_bp);
  buttons_layout->addWidget(cancel_bp);

  // THE WINDOW
  main_layout->addWidget(tab_widget);
  main_layout->addLayout(buttons_layout);

  setWindowTitle("Any settings you like");
  setLayout(main_layout);




  // check some checkbox if necessary
  QSettings stored_settings("egamh9", "mirrormachine");
  cb_detail_log->setChecked(
        stored_settings.value("detail_log", false).toBool());
  cb_add_collisions->setChecked(
        stored_settings.value("add_collisions", true).toBool());
  cb_generate_bsp->setChecked(
        stored_settings.value("generate_bsp", true).toBool());
  sb_max_leaf_size->setValue(
        stored_settings.value("max_leaf_size", 300).toUInt());
  cb_create_normals->setChecked(
        stored_settings.value("create_normals", true).toBool());
  cb_disable_bfc->setChecked(
        stored_settings.value("disable_bfc", false).toBool());
  cb_mirror_v_mapping->setChecked(
        stored_settings.value("mirror_v_mapping", true).toBool());
  cb_set_blp_extension->setChecked(
        stored_settings.value("set_blp_extension", true).toBool());
  cb_gray_missing_textures->setChecked(
        stored_settings.value("gray_missing_textures", true).toBool());
  cb_indoor->setChecked(
        stored_settings.value("indoor", false).toBool());
  cb_outdoor->setChecked(
        stored_settings.value("outdoor", true).toBool());
  le_custom_path->setText(
        stored_settings.value("custom_path", "").toString());

  cb_portals->setChecked(
        stored_settings.value("alpha_portals", false).toBool());
  cb_lights->setChecked(
        stored_settings.value("alpha_lights", false).toBool());
  cb_doodads->setChecked(
        stored_settings.value("alpha_doodads", false).toBool());
  cb_colors->setChecked(
        stored_settings.value("alpha_colors", false).toBool());
  cb_liquids->setChecked(
        stored_settings.value("alpha_liquids", false).toBool());
}


void SettingsWindow::closeSettings()
{
  this->close();
}


void SettingsWindow::updateSettings()
{
  QSettings stored_settings("egamh9", "mirrormachine");
  stored_settings.setValue("detail_log",
                           cb_detail_log->isChecked());
  stored_settings.setValue("add_collisions",
                           cb_add_collisions->isChecked());
  stored_settings.setValue("generate_bsp",
                           cb_generate_bsp->isChecked());
  stored_settings.setValue("max_leaf_size",
                           sb_max_leaf_size->value());
  stored_settings.setValue("create_normals",
                           cb_create_normals->isChecked());
  stored_settings.setValue("disable_bfc",
                           cb_disable_bfc->isChecked());
  stored_settings.setValue("mirror_v_mapping",
                           cb_mirror_v_mapping->isChecked());
  stored_settings.setValue("set_blp_extension",
                           cb_set_blp_extension->isChecked());
  stored_settings.setValue("gray_missing_textures",
                           cb_gray_missing_textures->isChecked());
  stored_settings.setValue("indoor",
                           cb_indoor->isChecked());
  stored_settings.setValue("outdoor",
                           cb_outdoor->isChecked());
  stored_settings.setValue("custom_path",
                           le_custom_path->text());

  stored_settings.setValue("alpha_portals",
                           cb_portals->isChecked());
  stored_settings.setValue("alpha_lights",
                           cb_lights->isChecked());
  stored_settings.setValue("alpha_doodads",
                           cb_doodads->isChecked());
  stored_settings.setValue("alpha_colors",
                           cb_colors->isChecked());
  stored_settings.setValue("alpha_liquids",
                           cb_liquids->isChecked());
  closeSettings();
}
