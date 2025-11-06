module group6_cs8050_assignment3 {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.desktop;

    exports group6_cs8050_assignment3;
    opens group6_cs8050_assignment3 to javafx.fxml;
}