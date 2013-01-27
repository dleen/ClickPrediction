name := "ClickPrediction Leen"

version := "1.0"

scalacOptions ++= Seq("-deprecation", "-unchecked")

libraryDependencies  ++= Seq(
            // other dependencies here
            // pick and choose:
            "org.scalanlp" %% "breeze-math" % "0.1",
            //"org.scalanlp" %% "breeze-learn" % "0.1",
            //"org.scalanlp" %% "breeze-process" % "0.1",
            //"org.scalanlp" %% "breeze-viz" % "0.1",
            "colt" % "colt" % "1.2.0"
		)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.2-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
            "Sonatype tools" at "https://oss.sonatype.org/content/groups/scala-tools/",
            "releases"  at "http://oss.sonatype.org/content/repositories/releases"
            )

mainClass in (Compile, run) := Some("main.scala.sgd.SGD")