package org.example.Trees;
//Auther: Abdelnasser Ouda
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextArea;
import javafx.stage.Stage;
import javafx.scene.layout.BorderPane;
import javafx.scene.control.*;
import javafx.scene.image.Image;

public class TreeVisualizerApp extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    private TextArea outputArea;

    private TreeView<String> tree24View;

  @Override
  public void start(Stage primaryStage) {
      TreeVisualizerController controller = new TreeVisualizerController();
      controller.setStage(primaryStage);  // Set the stage for the controller

      BorderPane root = new BorderPane();
      root.setCenter(controller.getView());
      root.setTop(createMenuBar(controller));

      Scene scene = new Scene(root, 1024, 1024);
      primaryStage.setTitle("Tree Visualizer");
      primaryStage.getIcons().add(new Image("file:src/main/resources/UMIcon.png"));
      primaryStage.setScene(scene);
      primaryStage.show();

      }

    private MenuBar createMenuBar(TreeVisualizerController controller) {
        MenuBar menuBar = new MenuBar();

        Menu fileMenu = new Menu("File");
        MenuItem saveItem = new MenuItem("Save Tree");
        MenuItem loadItem = new MenuItem("Load Tree");
        MenuItem exitItem = new MenuItem("Exit");

        saveItem.setOnAction(e -> controller.saveTree());
        loadItem.setOnAction(e -> controller.loadTree());
        exitItem.setOnAction(e -> System.exit(0));

        fileMenu.getItems().addAll(saveItem, loadItem, new SeparatorMenuItem(), exitItem);
        menuBar.getMenus().add(fileMenu);

        return menuBar;
    }
}